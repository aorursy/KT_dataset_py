import numpy as np

import sys

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook as tq

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report,make_scorer

from sklearn.cluster import KMeans
train = pd.read_csv('../input/eval-lab-3-f464/train.csv')

test = pd.read_csv('../input/eval-lab-3-f464/test.csv')
label_cols = ['gender','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription','PaymentMethod']
train = pd.get_dummies(data = train,columns = label_cols )

train.info()
test = pd.get_dummies(data = test,columns = label_cols )

test.info()
train.loc[train.TotalCharges==' ','TotalCharges']=0
train['TotalCharges']=train['TotalCharges'].astype(np.float64)
test.loc[test.TotalCharges==' ','TotalCharges']=0

test['TotalCharges']=test['TotalCharges'].astype(np.float64)
train2 = train.drop(['custId'],axis=1)

test2 = test.drop(['custId'],axis=1)
num = list(train2.columns)

num.remove('Satisfied')

ss = StandardScaler()

strain = train2.copy()

stest = test2.copy()

strain[num]=ss.fit_transform(train2[num])

stest[num]=ss.transform(test2[num])
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

X_lda = lda.fit_transform(strain.drop(['Satisfied'],axis=1), train['Satisfied'])
len(X_lda)
test_lda = pd.DataFrame(lda.transform(stest))

stest['lda_0'] = test_lda[0]
tr_lda = pd.DataFrame(lda.transform(strain.drop(['Satisfied'],axis=1)))

strain['lda_0'] = tr_lda[0]
strain = strain[['lda_0','Satisfied']]

stest = stest[['lda_0']]
# Preprocessing for second model

'''

strain = strain[['MonthlyCharges','TotalCharges','lda_0','tenure','Satisfied']]

stest = stest[['MonthlyCharges','TotalCharges','lda_0','tenure']]

'''
# Visualise Data for second model

'''

model=PCA(n_components=2)

model_data = model.fit(strain.drop('Satisfied',axis=1)).transform(strain.drop('Satisfied',axis=1))

plt.figure(figsize=(8,6))

plt.xlabel('X')

plt.ylabel('Y')

plt.title('Fig 1. PCA Representation of Given Classes')

plt.legend()

plt.scatter(model_data[:,0],model_data[:,1],label = strain['Satisfied'],c=strain['Satisfied'],cmap =plt.get_cmap('BrBG'))

plt.show()

'''
# Visualise Data for second model

'''

from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)

pca1.fit(strain.drop(['Satisfied'],axis=1))

T1 = pca1.transform(strain.drop(['Satisfied'],axis=1))

'''
# Visualise Data for second model

'''

wcss = []

for i in range(2, 20):

    kmean = KMeans(n_clusters = i, random_state = 47)

    kmean.fit(strain.drop(['Satisfied'],axis=1))

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,20),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()

'''
# Visualise Data for second model

'''

plt.figure(figsize=(16, 8))

preds1 = []

for i in range(2, 11):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(strain.drop(['Satisfied'],axis=1))

    pred = kmean.predict(strain.drop(['Satisfied'],axis=1))

    preds1.append(pred)

    

    plt.subplot(2, 5, i - 1)

    plt.title(str(i)+" clusters")

    plt.scatter(T1[:, 0], T1[:, 1], c=pred)

    

    centroids = kmean.cluster_centers_

    centroids = pca1.transform(centroids)

    plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)

    '''
strain.info()
stest.info()
# Parameter Tuning for clusters (Found clusters=8)

'''

from tqdm import tqdm_notebook as tq



mx, mn, mapping = 0, 0, 0, 0

y_true = train['Satisfied']



for num in tq(range(2, 18)):

    print('Iteration Start:', num)

    kmeans = KMeans(n_clusters = num, random_state=42)

    pred = kmeans.fit_predict(strain.drop(['Satisfied'],axis=1))

    

    for j in range(2**num):

        pmap = []

        for p in pred:

             pmap.append(((2**p)&j)>0)



        sc = roc_auc_score(y_true,pmap)

        if (sc>mx):

            mx = sc

            mn = num

            mapping = j

    print('Iteration End:', mn, mx, mapping)

    

print('Found optimal hyperparameters ->')

print('Number of clusters: ', mn)

print('Mapping: ', mapping)

'''
# Parameter Tuning for random state (centroid initialisation) (Found random state=42, mapping=154)



'''

from tqdm import tqdm_notebook as tq



mx, mr, mapping = 0, 0, 0, 0

y_true = train['Satisfied']



# Random centroid initialization and n_clusters simulation

for num in tq(range(50)):

    print('Iteration Start:', num)

    kmeans = KMeans(n_clusters = 8, random_state=num)

    pred = kmeans.fit_predict(strain.drop(['Satisfied'],axis=1))

    

    for j in range(2**8):

        pmap = []

        for p in pred:

            pmap.append(((2**p)&j)>0)



        sc = roc_auc_score(y_true,pmap)

        if (sc>mx):

            mx = sc

            mr = num

            mapping = j

    print('Iteration End:', mx, mr, mapping)

    

print('Found optimal hyperparameters ->')

print('Random State: ', mr)

print('Mapping: ', mapping)

'''
# Second prediction model after parameter tuning

'''

kmeans = KMeans(n_clusters = 12, random_state=42)

mapping = 3510

pred = kmeans.fit_predict(strain.drop(['Satisfied'],axis=1))

plt.figure(figsize=(16, 8))

plt.scatter(T1[:, 0], T1[:, 1], c=pred)

'''
kmeans = KMeans(n_clusters = 8, random_state=42)

mapping = 154

pred = kmeans.fit_predict(strain.drop(['Satisfied'],axis=1))
predK = kmeans.predict(stest)

fpred = []

for p in predK:

    fpred.append(((2**p)&mapping)>0)

fpred
df_final = pd.read_csv('../input/eval-lab-3-f464/test.csv')
df_final = df_final.join(pd.DataFrame(fpred))
df_final = df_final[['custId',0]]

df_final.info()
df_final = df_final.rename(columns={0: 'Satisfied'})
df_final['Satisfied'] = df_final['Satisfied'].astype(np.int64)
df_final.groupby(['Satisfied']).count()
df_final.head()
df_final.to_csv('final_sub.csv',index=False)