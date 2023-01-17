import os

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.manifold import TSNE

import seaborn as sns

%matplotlib inline



from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.cross_validation import KFold;



training_data = pd.read_csv('../input/preproc2_train.csv')

testing_data = pd.read_csv('../input/preproc2_test.csv')
training_data.head(1)
X = training_data.drop(['PassengerId','Survived'], axis=1)

#X = X.values

y = training_data['Survived']

X_t = testing_data.drop(['PassengerId'], axis=1)
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)

X_t_std = StandardScaler().fit_transform(X_t)
#testing new visu-tool to have a first feeling on correlations : 

X.plot(y='Fare',x='CabinFloor',kind='hexbin',gridsize=40,sharex=False, colormap='cubehelix', title='Hexbin of Survived and Age',figsize=(4,3))
cov_matrix = np.cov(X_std.T)

print('Covariance Matrix : %s' % cov_matrix)
e_vals, e_vecs = np.linalg.eig(cov_matrix)

print('EigenValues : %s' % e_vals)
for ev in e_vecs:

    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

print('Everything ok!')



# Make a list of (eigenvalue, eigenvector) tuples

e_pairs = [(np.abs(e_vals[i]), e_vecs[:,i]) for i in range(len(e_vals))]



# Sort the (eigenvalue, eigenvector) tuples from high to low

e_pairs.sort(key=lambda x: x[0], reverse=True)



# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in descending order:')

for i in e_pairs:

    print(i[0])



tot = sum(e_vals)

var_exp = [(i / tot)*100 for i in sorted(e_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)

with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(12, 8))



    plt.bar(range(12), var_exp, alpha=0.5, align='center',

            label='individual explained variance')

    plt.step(range(12), cum_var_exp, where='mid',

             label='cumulative explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
projection_matrix = np.hstack((e_pairs[0][1].reshape(12,1),

                             e_pairs[1][1].reshape(12,1),

                             e_pairs[2][1].reshape(12,1),

                             e_pairs[3][1].reshape(12,1),

                             e_pairs[4][1].reshape(12,1),

                             e_pairs[5][1].reshape(12,1),

                             e_pairs[6][1].reshape(12,1),

                             e_pairs[7][1].reshape(12,1)))

print(projection_matrix)

                          
X_proj = X_std.dot(projection_matrix)
plt.figure(figsize = (5,4))

plt.scatter(X_proj[:,0],X_proj[:,1], c='goldenrod',alpha=0.5)

plt.ylim(-10,10)

plt.show()
# Set a 2 KMeans clustering

kmeans = KMeans(n_clusters=2)

# Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(X_proj)



# Define our own color map

LABEL_COLOR_MAP = {0 : 'r',1 : 'b',2 : 'y'}

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]



# Plot the scatter digram

plt.figure(figsize = (7,7))

plt.scatter(X_proj[:,0],X_proj[:,1], c= label_color, alpha=0.5) 

plt.show()



df = pd.DataFrame(X_proj)

df['X_clustered']= X_clustered

sns.pairplot(df, hue='X_clustered', palette= 'Dark2', diag_kind='kde',size=1.85)

df.drop(['X_clustered'], axis=1)

df['label']=y

sns.pairplot(df, hue='label', palette= 'Dark2', diag_kind='kde')
ntrain = X_proj.shape[0]

SEED = 0 # for reproducibility
class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def feature_importances(self,x,y):

        print(self.clf.fit(x,y).feature_importances_)
from scipy.stats import randint

from sklearn.model_selection import RandomizedSearchCV

def hypertuning_rscv(est, p_distr, nbr_iter,X,y):

    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr,

                                  n_jobs=-1, n_iter=nbr_iter, cv=9)

    #CV = Cross-Validation ( here using Stratified KFold CV)

    start = time()

    rdmsearch.fit(X,y)

    print('hyper-tuning time : %d seconds' % (time()-start))

    start = 0

    ht_params = rdmsearch.best_params_

    ht_score = rdmsearch.best_score_

    return ht_params, ht_score
est = SVC()

from time import time

from scipy.stats import norm

svc_p_dist={'kernel':['linear','poly','rbf'],

            'C':norm(loc=0.5, scale=0.15)}

svc_parameters, svc_ht_score = hypertuning_rscv(est, svc_p_dist, 200, X_proj, y)

print(svc_parameters)

print('Hyper-tuned model score :')

print(svc_ht_score*100)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_parameters)

svc.train(X_proj,y)

svc.fit(X_proj,y)

pred = svc.predict(X_proj)

print(accuracy_score(pred, y)*100)
test_X_proj = X_t_std.dot(projection_matrix)

test_pred = svc.predict(test_X_proj)
PassengerId_test = testing_data['PassengerId']

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId_test,

                            'Survived': test_pred })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)
PassengerId_train = training_data['PassengerId']

output_X_proj = pd.DataFrame(X_proj)

output_X_proj['Survived']=y

output_X_proj['PassengerId']=PassengerId_train

output_test_X_proj = pd.DataFrame(test_X_proj)

output_test_X_proj['PassengerId'] = PassengerId_test

output_X_proj.to_csv('preproc3_train.csv', index = False)

output_test_X_proj.to_csv('preproc3_test.csv', index = False)
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0) # first try without init='pca'

X_tsne = tsne.fit_transform(X)

xtsne = pd.DataFrame(X_tsne)

xtsne['label']=y

xtsne.info()
c_map = {0:'b', 1:'r'}

plt.figure(figsize = (10,9))

plt.scatter(X_tsne[:,0],X_tsne[:,1], c=[c_map[_] for _ in xtsne['label']],alpha=0.5)

#plt.ylim(-10,10)

plt.show()
tsne_bis = TSNE(n_components=2, random_state=0, init='pca')

X_tsne_bis = tsne_bis.fit_transform(X)

xtsne_b = pd.DataFrame(X_tsne_bis)

xtsne_b['label']=y

c_map = {0:'b', 1:'r'}

fig = plt.figure(figsize = (10,9))

plt.scatter(X_tsne_bis[:,0],X_tsne_bis[:,1], c=[c_map[_] for _ in xtsne_b['label']],alpha=0.5)

plt.show()
tsne_ter = TSNE(n_components=3, random_state=0)#, init='pca')

X_tsne_ter = tsne_ter.fit_transform(X)

xtsne_t = pd.DataFrame(X_tsne_ter)

xtsne_t['label'] = y

X_tsne_test =tsne_ter.fit_transform(X_t)

xtsne_t_test = pd.DataFrame(X_tsne_test)
from mpl_toolkits.mplot3d import Axes3D

Axes3D

c_map = {0:'b', 1:'r'}

fig = plt.figure(figsize = (40,20))

ax = fig.add_subplot(251,projection='3d')

ax.scatter(xs=xtsne_t[0],ys=xtsne_t[1],zs=xtsne_t[2], c=[c_map[_] for _ in xtsne_t['label']])

plt.show()