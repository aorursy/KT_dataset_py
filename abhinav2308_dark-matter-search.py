# Sample solution is based on nontebooks by Artem Borzov



import numpy as np

import pandas as pd

import seaborn as sns

import scipy

import tables as tb

from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

from sklearn.neighbors import BallTree, KDTree, DistanceMetric

import glob



%pylab inline
# test = pd.read_hdf('../input/darkmatter/test.h5/test.h5')
# test = test.reset_index(drop=True)
def shortenDataSize(df):

    angleCols= [c for c in df.columns if ('TX' in c) or ('TY' in c)]

    for a in angleCols:

        df[a]= df[a].astype(np.float16)

    df['signal']= df['signal'].astype(np.int8)

    return df
feat_XY = ['X', 'Y']

def add_neighbours(df, k, metric='minkowski'):

    res = []

    

    for data_ind in tqdm(np.unique(df.data_ind)):

        ind = df.loc[df.data_ind == data_ind].copy()

        #как будет замечено, 1293 - это расстояние между слайсами по Z

        ind[['TX', 'TY']] *= 1293

        values = np.unique(ind.Z)

        

        for j in range(1, len(values)):

            z, z_next = (ind.loc[ind.Z == values[j-1]].copy(),

                         ind.loc[ind.Z == values[j]].copy())

            

            b_tree = BallTree(z_next[feat_XY], metric=metric)

            d, i = b_tree.query(z[feat_XY], k=min(k, len(z_next)))

            

            for m in range(i.shape[1]):

                data = z_next.iloc[i[:, m]]

                z_copy = z.copy()

                for col in feat_XY:

                    z_copy[col + '_pair_'+str(m)] = data[col].values

                    

                res.append(z_copy)

            

        res.append(z_next)

        

    res = pd.concat(res)

    pairCols= [c for c in res.columns if 'pair' in c]

    for p in pairCols:

        col= p.split('_')[0]

        res['d' + p] = res[col].values - res[p].values

    res.drop(pairCols,axis=1,inplace=True)

    return res



def balance_train(df, k):

    df= shortenDataSize(df)

    data = add_neighbours(df, k=k)

    noise = data.event_id == -999

    signal, not_signal = data.loc[np.logical_not(noise)], data.loc[noise]

    noise_part = not_signal.sample(len(signal))

    return pd.concat([signal, noise_part]).reset_index(drop=True)

%%time

train = []

for file in glob.glob('../input/dark-matter-search/training/open*.h5')[:5]: # just 5 bricks

    train.append(balance_train(pd.read_hdf(file), k=5))

train = pd.concat(train)
# train.info()
train= shortenDataSize(train)
# np.finfo(numpy.float16).max
# train.describe().T
y_train = train.signal

X_train = train.drop(['event_id', 'signal', 'data_ind'], axis=1)

del train
# X_train.describe().T
from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import StratifiedKFold, GridSearchCV,train_test_split
model = CatBoostClassifier(iterations=100,

                           learning_rate=0.1,

                           loss_function='Logloss',

                           verbose=True)
%%time

model.fit(X_train, y_train)
feature_score = pd.DataFrame(list(zip(X_train.dtypes.index, model.get_feature_importance(Pool(X_train, label=y_train)))),

                columns=['Feature','Score'])



feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')
plt.rcParams["figure.figsize"] = (12,7)

ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')

ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)

ax.set_xlabel('')



rects = ax.patches



labels = feature_score['Score'].round(2)



for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')



plt.show()
# prepared_test = add_neighbours(test, k=5)

# X_test = prepared_test.drop(['data_ind'], axis=1)

# X_test= shortenDataSize(X_test)
# probas = model.predict_proba(X_test)[:,1]
# probas
df = pd.DataFrame({'id': prepared_test.index, 'signal': probas}).groupby('id')

agg = df.aggregate(('mean')).loc[:, ['signal']]
# agg.shape
# agg.head(5)
# from IPython.display import FileLink
# agg.to_csv('submission.csv.gz', index=True, compression='gzip')
# FileLink('submission.csv.gz')