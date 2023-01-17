import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import plotly.offline as py

py.init_notebook_mode(connected = True)

import plotly.graph_objs as go

import plotly.tools as tls

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



from sklearn.decomposition import PCA
train = pd.read_csv('/kaggle/input/cervical-cancer-risk-classification/kag_risk_factors_cervical_cancer.csv')

train.head()
print(train.shape)
print(train.dtypes)
for i in range(1,28):

    train.iloc[:,i]=pd.to_numeric(train.iloc[:,i], errors='coerce')

print(train.dtypes)
train.isnull().sum()
train2 = train[train.columns[28:37]]

train2.reset_index(drop=True, inplace=True)

train3= train[["Age"]]

train3.reset_index(drop=True, inplace=True)

train4=train[["STDs: Number of diagnosis"]]

train4.reset_index(drop=True, inplace=True)

frames=[train2,train3,train4]

first_PCA = pd.concat(frames, axis=1)

first_PCA.head()
from sklearn.decomposition import PCA

pca = PCA(n_components=6)

pca.fit(first_PCA)

pca_output = pca.transform(first_PCA)

ps = pd.DataFrame(pca_output)

ps.head()
print(pca.explained_variance_ratio_)
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import proj3d

two_comp = pd.DataFrame(ps[[0,1]])



fig = plt.figure(figsize=(8,8))

plt.plot(two_comp[0], two_comp[1], 'x', markersize=6, color='blue', alpha=0.5)





plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component')

plt.show()
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



clusterer = KMeans(n_clusters=3,random_state=42).fit(two_comp)

centers = clusterer.cluster_centers_

c_preds = clusterer.predict(two_comp)



fig = plt.figure(figsize=(8,8))

colors = ['orange','blue','green']

colored = [colors[k] for k in c_preds]



plt.scatter(two_comp[0],two_comp[1],  color = colored)

for i,c in enumerate(centers):

    plt.plot(c[0], c[1], 'X', markersize=10, color='red', alpha=0.9, label=''+str(i))



plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component')

plt.legend()

plt.show()
first_PCA['cluster']=c_preds

first_PCA.head(10)
fig = plt.figure(figsize=(8,8))

plt.plot(first_PCA['cluster'], first_PCA['Age'], 'x', markersize=6, color='blue')



plt.xlabel('Cluster')

plt.ylabel('Age')

plt.show()
#Continuous variable imputation

train['Number of sexual partners'].fillna(train['Number of sexual partners'].median(), inplace=True)

train['First sexual intercourse'].fillna(train['First sexual intercourse'].median(), inplace=True)

train['Num of pregnancies'].fillna(train['Num of pregnancies'].median(), inplace=True)

train['Smokes (years)'].fillna(train['Smokes (years)'].median(), inplace=True)

train['Smokes (packs/year)'].fillna(train['Smokes (packs/year)'].median(), inplace=True)

train['Hormonal Contraceptives (years)'].fillna(train['Hormonal Contraceptives (years)'].median(), inplace=True)

train['STDs (number)'].fillna(train['STDs (number)'].median(), inplace=True)

train['STDs: Time since first diagnosis'].fillna(train['STDs: Time since first diagnosis'].median(), inplace=True)

train['STDs: Time since last diagnosis'].fillna(train['STDs: Time since last diagnosis'].median(), inplace=True)

train['IUD (years)'].fillna(train['IUD (years)'].median(), inplace=True)



#Discrete variable imputation (without population estimates)

train['STDs:condylomatosis'].fillna(train['STDs:condylomatosis'].median(), inplace=True)

train['STDs:cervical condylomatosis'].fillna(train['STDs:cervical condylomatosis'].median(), inplace=True)

train['STDs:vaginal condylomatosis'].fillna(train['STDs:vaginal condylomatosis'].median(), inplace=True)

train['STDs:vulvo-perineal condylomatosis'].fillna(train['STDs:vulvo-perineal condylomatosis'].median(), inplace=True)

train['STDs:syphilis'].fillna(train['STDs:syphilis'].median(), inplace=True)

train['STDs:molluscum contagiosum'].fillna(train['STDs:molluscum contagiosum'].median(), inplace=True)

train['STDs'].fillna(train['STDs'].median(), inplace=True)

train['STDs:AIDS'].fillna(train['STDs:AIDS'].median(), inplace=True)

train['STDs:HIV'].fillna(train['STDs:HIV'].median(), inplace=True)
#Discrete variable imputation (with population estimates)

s = int(.136*train.shape[0])

smokes = np.hstack((np.ones(s), np.zeros(train.shape[0]-s)))

np.random.shuffle(smokes)

train['Smokes'].fillna(pd.Series(smokes), axis=0, inplace=True)



s = int(.103*train.shape[0])

iud = np.hstack((np.ones(s), np.zeros(train.shape[0]-s)))

np.random.shuffle(iud)

train['IUD'].fillna(pd.Series(iud), axis=0, inplace=True)



s = int(.229*train.shape[0])

hc = np.hstack((np.ones(s), np.zeros(train.shape[0]-s)))

np.random.shuffle(hc)

train['Hormonal Contraceptives'].fillna(pd.Series(hc), axis=0, inplace=True)



s = int(.399*train.shape[0])

hpv = np.hstack((np.ones(s), np.zeros(train.shape[0]-s)))

np.random.shuffle(hpv)

train['STDs:HPV'].fillna(pd.Series(hpv), axis=0, inplace=True)



s = int(.034*train.shape[0])

hep = np.hstack((np.ones(s), np.zeros(train.shape[0]-s)))

np.random.shuffle(hep)

train['STDs:Hepatitis B'].fillna(pd.Series(hep), axis=0, inplace=True)



s = int(.159*train.shape[0])

gen = np.hstack((np.ones(s), np.zeros(train.shape[0]-s)))

np.random.shuffle(gen)

train['STDs:genital herpes'].fillna(pd.Series(gen), axis=0, inplace=True)



s = int(.044*train.shape[0])

pid = np.hstack((np.ones(s), np.zeros(train.shape[0]-s)))

np.random.shuffle(pid)

train['STDs:pelvic inflammatory disease'].fillna(pd.Series(pid), axis=0, inplace=True)
train.isnull().sum()
pca = PCA(n_components=6)

pca.fit(train)

pca_output2 = pca.transform(train)

ps2 = pd.DataFrame(pca_output2)

ps2.head()
print(pca.explained_variance_ratio_)
two_comp = pd.DataFrame(ps2[[0,1]])

fig = plt.figure(figsize=(8,8))

plt.plot(two_comp[0], two_comp[1], 'x', markersize=6, color='blue')



plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component')

plt.show()
tocluster = pd.DataFrame(ps2[[0,1]])

clusterer = KMeans(n_clusters=4,random_state=42).fit(tocluster)

centers = clusterer.cluster_centers_

c_preds = clusterer.predict(tocluster)



fig = plt.figure(figsize=(8,8))

colors = ['orange','blue','green','purple']

colored = [colors[k] for k in c_preds]



plt.scatter(two_comp[0],two_comp[1],  color = colored)

for i,c in enumerate(centers):

    plt.plot(c[0], c[1], 'X', markersize=10, color='red', alpha=0.9, label=''+str(i))



plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component')

plt.legend()

plt.show()
train['cluster']=c_preds



fig = plt.figure(figsize=(8,8))

plt.plot(train['cluster'], train['Age'], 'x', markersize=6, color='blue')



plt.xlabel('Cluster')

plt.ylabel('Age')

plt.show()