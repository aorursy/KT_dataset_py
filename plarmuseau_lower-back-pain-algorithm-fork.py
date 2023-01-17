import pandas as pd

import numpy as np

import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline



# read data into dataset variable

train = pd.read_csv("../input/Dataset_spine.csv")





# Drop the unnamed column in place (not a copy of the original)#

train.drop('Unnamed: 13', axis=1, inplace=True)

train.columns = ['Pelvic Incidence','Pelvic Tilt','Lumbar Lordosis Angle','Sacral Slope','Pelvic Radius', 'Spondylolisthesis Degree', 'Pelvic Slope', 'Direct Tilt', 'Thoracic Slope', 'Cervical Tilt','Sacrum Angle', 'Scoliosis Slope','Outcome']

# Concatenate the original df with the dummy variables

#data = pd.concat([data, pd.get_dummies(data['Class_att'])], axis=1)



# Drop unnecessary label column in place. 

#data.drop(['Class_att','Normal'], axis=1, inplace=True)

print(train)

train.describe().T
# Categorical features

cat_cols = []

for c in train.columns:

    if train[c].dtype == 'object':

        cat_cols.append(c)

print('Categorical columns:', cat_cols)



# Dublicate features

d = {}; done = []

cols = train.columns.values

for c in cols: d[c]=[]

for i in range(len(cols)):

    if i not in done:

        for j in range(i+1, len(cols)):

            if all(train[cols[i]] == train[cols[j]]):

                done.append(j)

                d[cols[i]].append(cols[j])

dub_cols = []

for k in d.keys():

    if len(d[k]) > 0: 

        # print k, d[k]

        dub_cols += d[k]        

print('Dublicates:', dub_cols)



# Constant columns

const_cols = []

for c in cols:

    if len(train[c].unique()) == 1:

        const_cols.append(c)

print('Constant cols:', const_cols)
def add_new_col(x):

    if x not in new_col.keys(): 

        # set n/2 x if is contained in test, but not in train 

        # (n is the number of unique labels in train)

        # or an alternative could be -100 (something out of range [0; n-1]

        return int(len(new_col.keys())/2)

    return new_col[x] # rank of the label



def clust(x):

    kl=0

    if x<0.75:

        kl=1

    if x>0.75 and x<4:

        kl=2

    if x>4:

        kl=4

    return kl



new_col= train[['Pelvic Tilt','Outcome']].groupby('Outcome').describe().fillna(method='bfill')

new_col.columns=['count','mean','std','min','p25','p50','p75','max']

new_col['eff']=new_col['std']/new_col['mean']

new_col['eff2']=new_col['eff']*new_col['std']

new_col['clust']=new_col['eff2'].map(clust)
print(new_col)

print(train)
train=pd.merge(train,new_col, how='inner', left_on='Outcome', right_index=True)

sns.pairplot(train[['Pelvic Tilt','std','eff2','Outcome']],hue='Outcome')

plt.show()
from sklearn.decomposition import PCA, FastICA

from sklearn.decomposition import TruncatedSVD

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection

from sklearn.cluster import KMeans



from mpl_toolkits.mplot3d import Axes3D





import seaborn as sns

from pandas.plotting import scatter_matrix

# INPUT df  (dataframe en welke kolommen je gebruikt om te klusteren)

# define 'clust' groep

# define drop colomns



#-------------------------------------

labels= train['clust']

X = train.drop('Outcome',axis=1)

n_comp = 5  #define number of clusters

#-------------------------------------



print('-------Principal Component Analysis---------')

# PCA

pca = PCA(n_components=n_comp, random_state=4)

results = pca.fit_transform(X)

results=pd.DataFrame(results)

results['clust']=labels

sns.set(style="ticks")

sns.pairplot(results,hue='clust')

plt.show()

# To getter a better understanding of interaction of the dimensions

# plot the first three PCA dimensions

fig = plt.figure(1, figsize=(12, 12))

ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(results[0], results[1], results[2], c=labels, cmap=plt.cm.Paired)

ax.set_title("First three Singular Value")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])



print('-------Singular Value Decomposition---------')

# tSVD

tsvd = TruncatedSVD(n_components=n_comp, random_state=420)

results = tsvd.fit_transform(X)

results=pd.DataFrame(results)

results['clust']=labels

sns.set(style="ticks")

sns.pairplot(results,hue='clust')

plt.show()



print('-------Fast I  Component Analysis---------')

# ICA

ica = FastICA(n_components=n_comp, random_state=420)

results = ica.fit_transform(X)

results=pd.DataFrame(results)

results['clust']=labels

sns.set(style="ticks")

sns.pairplot(results,hue='clust')

plt.show()



print('-------Gaussian Random Projection---------')

# GRP

grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)

results = grp.fit_transform(X)

results=pd.DataFrame(results)

results['clust']=labels

sns.set(style="ticks")

sns.pairplot(results,hue='clust')

plt.show()



# To getter a better understanding of interaction of the dimensions

# plot the first three PCA dimensions

fig = plt.figure(1, figsize=(12, 12))

ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(results[0], results[1], results[2], c=labels, cmap=plt.cm.Paired)

ax.set_title("First three Gaussian")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])



print('-------Sparse Random Projection---------')

# SRP

srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)

results = srp.fit_transform(X)

results=pd.DataFrame(results)

results['clust']=labels

sns.set(style="ticks")

sns.pairplot(results,hue='clust')

plt.show()





plt.show()





print('-------KMeans Classification---------')

#Kmeans

kmeans = KMeans(n_clusters=n_comp, random_state=0).fit(X)

results=kmeans.transform(X)

results=pd.DataFrame(results)

results['clust']=labels

sns.set(style="ticks")

sns.pairplot(results,hue='clust')

plt.show()





# To getter a better understanding of interaction of the dimensions

# plot the first three PCA dimensions

fig = plt.figure(1, figsize=(12, 12))

ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(results[0], results[1], results[2], c=labels, cmap=plt.cm.Paired)

ax.set_title("First three Kmeanss")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])



#adding the cluster that best separates the variability

# input X=df[kolom] from previous

# input df dataframe

print('-------PCA---------')

# PCA

# PCA

pca = PCA(n_components=n_comp, random_state=4)

results = pca.fit_transform(X)

results=pd.DataFrame(results)

results['clust']=labels







#print(results)

from mpl_toolkits.mplot3d import Axes3D

# To getter a better understanding of interaction of the dimensions

# plot the first three PCA dimensions

fig = plt.figure(1, figsize=(12, 12))

ax = Axes3D(fig, elev=-20, azim=85)

ax.scatter(results[0], results[1], results[2], c=labels, cmap=plt.cm.Paired)

ax.set_title("First three Sparse Random Projections")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])





#Append decomposition components to datasets  # to do in next part

for i in range(1, n_comp + 1):

    train['pca_' + str(i)] = results[i - 1]

from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(train.drop('Outcome',axis=1).astype(np.float64),

    train.clust.astype(np.float64), train_size=0.75, test_size=0.25)



tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)

tpot.fit(X_train, y_train)

print(tpot.score(X_test, y_test))

tpot.export('tpot_iris_pipeline.py')
ypred = pl.predict(X_test)

ypred = ypred.reshape(-1,1)



pl.score(X_test, y_test)