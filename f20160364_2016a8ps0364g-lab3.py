import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from sklearn import cluster, mixture # For clustering

from sklearn.metrics import roc_auc_score

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch

from sklearn.preprocessing import StandardScaler, normalize, RobustScaler, MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score
df=pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

df1=pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")
training_data_labels = df['Satisfied']

df.loc[df['TotalCharges'] == " ", "TotalCharges"] = df["MonthlyCharges"]

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

df.head()
numerical_features = ['custId','tenure', 'MonthlyCharges', 'TotalCharges']

categorical_features = ['SeniorCitizen','gender', 'Married', 'Children','TVConnection', 'Channel1', 'Channel2','Channel3', 'Channel4', 'Channel5', 'Channel6', 'AddedServices','Internet','HighSpeed', 'Subscription', 'PaymentMethod']
from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

le = LabelEncoder()



df[categorical_features] = df[categorical_features].apply(lambda col: le.fit_transform(col))

df.head(10)
corr=df.corr()

plt.figure(figsize=(12,9))

mask = np.zeros_like(corr)

cmap=sns.diverging_palette(220,10,as_cmap=True)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

  ax = sns.heatmap(corr,cmap=cmap,mask=mask, vmax=1,vmin=-1, square=True)
def getmask(df):

    categorical_feature_mask = []

    for variable in df.columns:

        if variable in categorical_features:

            categorical_feature_mask.append(True)

        else:

            categorical_feature_mask.append(False)

    return categorical_feature_mask
df.drop(['Satisfied'],axis=1,inplace=True)

df=pd.concat([df[categorical_features],df[numerical_features]],axis=1)



from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features = getmask(df), sparse=False ) 

X_ohe = ohe.fit_transform(df)

X = pd.get_dummies(df, prefix_sep='_', drop_first=True)
import numpy as np

from kmodes.kmodes import KModes

from kmodes.kprototypes import KPrototypes



km = KModes(n_clusters=2, init='Huang',max_iter=1000000 ,n_init=5, verbose=1,random_state=0)

clusters1 = km.fit_predict(X_ohe)

(1-roc_auc_score(training_data_labels, clusters1))
categorical_cols = df1.columns[getmask(df1)].tolist()

from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()



df1[categorical_features] = df1[categorical_features].apply(lambda col: le1.fit_transform(col))



from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features = getmask(df1), sparse=False)

X_test = ohe.fit_transform(df)

X_test = pd.get_dummies(df1, prefix_sep='_', drop_first=True)
km = KModes(n_clusters=2, init='Huang',max_iter=1000000 ,n_init=5, verbose=1,random_state=0)

clusters = km.fit_predict(X_test)
df1['Satisfied']=np.array(clusters)

lis= [df1['custId']]

out = pd.DataFrame(data={'custId':df1['custId'],'Satisfied':df1['Satisfied']})

out.to_csv('Kmodes1.csv',index=False)
numerical_features2 = ['tenure', 'MonthlyCharges', 'TotalCharges']

categorical_features2 = ['SeniorCitizen', 'Married', 'Children','TVConnection', 'Channel1', 'Channel2','Channel3', 'Channel4', 'Channel5', 'Channel6', 'AddedServices', 'Subscription', 'PaymentMethod']
df2=pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

df12=pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")
df12.head()
from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

le2 = LabelEncoder()

df2.drop(['Satisfied'],axis=1,inplace=True)



df12[categorical_features] = df12[categorical_features].apply(lambda col: le2.fit_transform(col))

df2[categorical_features] = df2[categorical_features].apply(lambda col: le2.fit_transform(col))



df2=pd.concat([df2[categorical_features],df2[numerical_features]],axis=1)
from sklearn.preprocessing import OneHotEncoder

ohe2 = OneHotEncoder(categorical_features = getmask(df12[categorical_features]), sparse=False ) 

X2 = ohe2.fit_transform(df2[categorical_features])

X_test2 = pd.get_dummies(df12[categorical_features], prefix_sep='_', drop_first=True)

brc = Birch(threshold=0.5, branching_factor=50, n_clusters=2, compute_labels=True, copy=True).fit(X_test2)

pred = brc.labels_
df12['Satisfied']=np.array(pred)

lis= [df12['custId']]

out2 = pd.DataFrame(data={'custId':df12['custId'],'Satisfied':df12['Satisfied']})

out2.to_csv('Birch1.csv',index=False)