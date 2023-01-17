import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import glob
import os

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event
from IPython.display import display

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
train = np.unique([p.split('-')[0] for p in sorted(glob.glob('../input/train_*/**'))])
test = np.unique([p.split('-')[0] for p in sorted(glob.glob('../input/test/**'))])
# det = pd.read_csv('../input/detectors.csv')
# sub = pd.read_csv('../input/sample_submission.csv')

def read_files(path):
    return [p.split('-')[0] for p in sorted(glob.glob(path))]

train_file_names = np.unique(read_files('../input/train_*/**.csv'))
test_file_names = np.unique(read_files('../input/test/**.csv'))
detector = pd.read_csv('../input/detectors.csv')
submission = pd.read_csv('../input/sample_submission.csv')
hits, cells, particles, truth = load_event(train[0])
print(len(hits), len(cells), len(particles), len(truth))
n=10
print("hits:")
display(hits.head(n))
print("cells:")
display(cells.head(n))
print("particles:")
display(particles.head(n))
print("truth:")
display(truth.head(n))
denoise_model = Sequential()
denoise_model.add(Dense(256, activation='relu', input_shape=[25, ]))
denoise_model.add(Dense(256, activation='relu'))
denoise_model.add(Dense(256, activation='relu'))
denoise_model.add(Dropout(0.5))
denoise_model.add(Dense(256, activation='relu'))
denoise_model.add(Dense(256, activation='relu'))
denoise_model.add(Dense(256, activation='relu'))
denoise_model.add(BatchNormalization())
denoise_model.add(Dense(1, activation='sigmoid'))

denoise_model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
def get_train_data_for_denoise(e):
    # https://www.kaggle.com/meaninglesslives/classifier-hdbscan-helixfitting
    hits, cells, truth = load_event(e, parts=['hits', 'cells', 'truth'])
    hits['event_id'] = int(e[-9:])
    cells = cells.groupby(by=['hit_id'])['ch0', 'ch1', 'value'].agg(
        ['mean', 'median', 'max', 'min', 'sum']).reset_index()
    cells.columns = ['hit_id'] + ['-'.join([c2, c1]) for c1 in [
        'mean', 'median', 'max', 'min', 'sum']
        for c2 in ['ch0', 'ch1', 'value']]
    hits = pd.merge(hits, cells, how='left', on='hit_id')
    tcols = list(truth.columns)
    hits = pd.merge(hits, truth, how='left', on='hit_id')
    hits = norm_points(hits)
    cols = [c for c in hits.columns if c not in [
        'event_id', 'hit_id', 'particle_id'] + tcols]

    # noise marking
    # noise -> target==0
    # genuine data -> target==1
    hits['target'] = hits['particle_id'].map(lambda x: 0 if x == 0 else 1)

    # return x,y
    return hits[cols], hits['target'].values


# https://www.kaggle.com/mikhailhushchyn/dbscan-benchmark
# https://www.kaggle.com/mikhailhushchyn/hough-transform
def norm_points(df, g=0.7):
    x = df.x.values
    y = df.y.values
    z = df.z.values
    r = np.sqrt(x**2 + y**2 + z**2)
    df['x2'] = x / r
    df['y2'] = y / r
    r = np.sqrt(x**2 + y**2)
    df['z2'] = (z / r) * g

    df['r'] = r
    # df['r2'] = np.sqrt(x**2 + y**2)
    # df['r3'] = np.sqrt(x**2 + z**2)
    # df['r4'] = np.sqrt(y**2 + z**2)
    # df['phi'] = np.arctan2(y, x)
    # df['phi2'] = np.arctan2(y, z)
    # df['phi3'] = np.arctan2(x, z)
    # df['hm'] = (2. * np.cos(df['phi'] - g) / df['r2']).values
    return df

# denoise_train_batch_size = 4096
# denoise_train_epochs = 2

# for f in tqdm(train_file_names[0:2]):
#     X, y = get_train_data_for_denoise(f)
#     denoise_model.fit(X, y, batch_size=denoise_train_batch_size, 
#                       epochs=denoise_train_epochs)
# use sklearn's train_test_split
# proper_train_data = []
# noise_train_data = []

# for d in train_data:
# #     noise==0, proper==1
#     if denoise_model.predict(d) >= 0.5:
#         proper_train_data.append(d)
#     else:
#         noise_train_data.append(d)
import xgboost as xgb
from sklearn import preprocessing, model_selection
import hdbscan

scl = preprocessing.StandardScaler()
#dbscan = cluster.DBSCAN(eps=0.007555, min_samples=1, algorithm='kd_tree', n_jobs=-1)
dbscan = hdbscan.HDBSCAN(min_samples=3, 
                         min_cluster_size=5, 
                         cluster_selection_method='leaf', 
                         prediction_data=False, 
                         metric='braycurtis')
from sklearn.preprocessing import StandardScaler
import hdbscan
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import DBSCAN

class Clusterer(object):
    def __init__(self,rz_scales=[0.65, 0.965, 1.528]):                        
        self.rz_scales=rz_scales
    
    def _eliminate_outliers(self,labels,M):
        norms=np.zeros((len(labels)),np.float32)
        indices=np.zeros((len(labels)),np.float32)
        for i, cluster in tqdm(enumerate(labels),total=len(labels)):
            if cluster == 0:
                continue
            index = np.argwhere(self.clusters==cluster)
            index = np.reshape(index,(index.shape[0]))
            indices[i] = len(index)
            x = M[index]
            norms[i] = self._test_quadric(x)
        threshold1 = np.percentile(norms,90)*5
        threshold2 = 25
        threshold3 = 6
        for i, cluster in enumerate(labels):
            if norms[i] > threshold1 or indices[i] > threshold2 or indices[i] < threshold3:
                self.clusters[self.clusters==cluster]=0   
    def _test_quadric(self,x):
        if x.size == 0 or len(x.shape)<2:
            return 0
        Z = np.zeros((x.shape[0],10), np.float32)
        Z[:,0] = x[:,0]**2
        Z[:,1] = 2*x[:,0]*x[:,1]
        Z[:,2] = 2*x[:,0]*x[:,2]
        Z[:,3] = 2*x[:,0]
        Z[:,4] = x[:,1]**2
        Z[:,5] = 2*x[:,1]*x[:,2]
        Z[:,6] = 2*x[:,1]
        Z[:,7] = x[:,2]**2
        Z[:,8] = 2*x[:,2]
        Z[:,9] = 1
        v, s, t = np.linalg.svd(Z,full_matrices=False)        
        smallest_index = np.argmin(np.array(s))
        T = np.array(t)
        T = T[smallest_index,:]        
        norm = np.linalg.norm(np.dot(Z,T), ord=2)**2
        return norm

    def _preprocess(self, hits):
        
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values

        r = np.sqrt(x**2 + y**2 + z**2)
        hits['x2'] = x/r
        hits['y2'] = y/r

        r = np.sqrt(x**2 + y**2)
        hits['z2'] = z/r

        ss = StandardScaler()
        X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)
        for i, rz_scale in enumerate(self.rz_scales):
            X[:,i] = X[:,i] * rz_scale
       
        return X
    
    def _init(self,dfh):
        dfh['r'] = np.sqrt(dfh['x'].values**2+dfh['y'].values**2+dfh['z'].values**2)
        dfh['rt'] = np.sqrt(dfh['x'].values**2+dfh['y'].values**2)
        dfh['a0'] = np.arctan2(dfh['y'].values,dfh['x'].values)
        dfh['z1'] = dfh['z'].values/dfh['rt'].values
        dfh['x2'] = 1/dfh['z1'].values
        dz0 = -0.00070
        stepdz = 0.00001
        stepeps = 0.000005
        mm = 1
        for ii in tqdm(range(100)):
            mm = mm*(-1)
            dz = mm*(dz0+ii*stepdz)
            dfh['a1'] = dfh['a0'].values+dz*dfh['z'].values*np.sign(dfh['z'].values)
            dfh['sina1'] = np.sin(dfh['a1'].values)
            dfh['cosa1'] = np.cos(dfh['a1'].values)
            dfh['x1'] = dfh['a1'].values/dfh['z1'].values
            ss = StandardScaler()
            dfs = ss.fit_transform(dfh[['sina1','cosa1','z1','x1','x2']].values)
            cx = np.array([1, 1, 0.75, 0.5, 0.5])
            for k in range(5):
                dfs[:,k] *= cx[k]
            clusters=DBSCAN(eps=0.0035+ii*stepeps,min_samples=1,metric='euclidean',n_jobs=4).fit(dfs).labels_            
            if ii==0:
                dfh['s1'] = clusters
                dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
            else:
                dfh['s2'] = clusters
                dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
                maxs1 = dfh['s1'].max()
                cond = np.where((dfh['N2'].values>dfh['N1'].values) & (dfh['N2'].values<20))
                s1 = dfh['s1'].values
                s1[cond] = dfh['s2'].values[cond]+maxs1
                dfh['s1'] = s1
                dfh['s1'] = dfh['s1'].astype('int64')
                dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
        return dfh['s1'].values    
    def predict(self, hits):         
        self.clusters = self._init(hits) 
        X = self._preprocess(hits) 
        cl = hdbscan.HDBSCAN(min_samples=1,min_cluster_size=7,
                             metric='braycurtis',cluster_selection_method='leaf',algorithm='best', leaf_size=50)
        labels = np.unique(self.clusters)
        self._eliminate_outliers(labels,X)
        n_labels = 0
        while n_labels < len(labels):
            n_labels = len(labels)            
            max_len = np.max(self.clusters)
            mask = self.clusters == 0
            self.clusters[mask] = cl.fit_predict(X[mask])+max_len
        return self.clusters
model = Clusterer()
labels = model.predict(hits)
def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission
submission = create_one_event_submission(0, hits, labels)
score = score_event(truth, submission)
print("Your score: ", score)
dataset_submissions = []
dataset_scores = []
path_to_train = "../input/train_1"

for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=0, nevents=5):
    # Track pattern recognition
    model = Clusterer()
    labels = model.predict(hits)

    # Prepare submission for an event
    one_submission = create_one_event_submission(event_id, hits, labels)
    dataset_submissions.append(one_submission)

    # Score for the event
    score = score_event(truth, one_submission)
    dataset_scores.append(score)

    print("Score for event %d: %.8f" % (event_id, score))
print('Mean score: %.8f' % (np.mean(dataset_scores)))
path_to_test = "../input/test"
test_dataset_submissions = []

create_submission = True # True for submission 
if create_submission:
    for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):

        # Track pattern recognition 
        model = Clusterer()
        labels = model.predict(hits)

        # Prepare submission for an event
        one_submission = create_one_event_submission(event_id, hits, labels)
        test_dataset_submissions.append(one_submission)
        
        print('Event ID: ', event_id)

    # Create submission file
    submission = pd.concat(test_dataset_submissions, axis=0)
    submission.to_csv('submission.csv', index=False)