import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

from collections import Counter

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch



from sklearn.preprocessing import StandardScaler, normalize, RobustScaler, MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score

df = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

df_test = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

df.loc[df['TotalCharges'] == " ", "TotalCharges"] = df["MonthlyCharges"]

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

df.head()
corr=df.corr()

plt.figure(figsize=(12,9))

mask = np.zeros_like(corr)

cmap=sns.diverging_palette(220,10,as_cmap=True)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

  ax = sns.heatmap(corr,cmap=cmap,mask=mask, vmax=1,vmin=-1, square=True)
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

categorical_features = ['gender','TVConnection', 'Married', 'Children', 'Channel1','Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6', 'Internet','HighSpeed','AddedServices', 'Subscription', 'PaymentMethod']
from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

le = LabelEncoder()

df[categorical_features] = df[categorical_features].apply(lambda col: le.fit_transform(col))
sm = SMOTE()

X = pd.get_dummies(df.drop(['custId', 'gender','Satisfied','Internet','HighSpeed'], axis=1))

y = df['Satisfied']

X_res, y_res = sm.fit_resample(X, y)
scale = StandardScaler()

X = scale.fit_transform(X)

X = normalize(X)
scale = RobustScaler()

X_test = df_test.drop(['custId', 'gender','TVConnection','Internet','HighSpeed','SeniorCitizen','Children','Married'], axis=1)

X_test = pd.get_dummies(X_test)

X_test = X_test

X_test = scale.fit_transform(X_test)

X_test = normalize(X_test)
algo = Birch(n_clusters=2).fit(X_test)

labels = algo.labels_

1-labels
result = pd.DataFrame({'Satisfied':labels}, index=df_test['custId'])

result.to_csv("output.csv")
def getmask(df):

    categorical_feature_mask = []

    for variable in df.columns:

        if variable in categorical_features:

            categorical_feature_mask.append(True)

        else:

            categorical_feature_mask.append(False)

    return categorical_feature_mask
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features = getmask(df), sparse=False ) 

X_ohe = ohe.fit_transform(df) 

X = pd.get_dummies(df, prefix_sep='_', drop_first=True)
import numpy as np

from kmodes.kmodes import KModes

from kmodes.kprototypes import KPrototypes



km = KModes(n_clusters=2, init='Huang',max_iter=1000000 ,n_init=5, verbose=1,random_state=0)

clusters1 = km.fit_predict(X_ohe)

df1= pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

df1.fillna(value=df.mean(),inplace=True)

scaler = RobustScaler()
xencoded=pd.get_dummies(df1[categorical_features])

xencoded.head()
categorical_cols = df1.columns[getmask(df1)].tolist()

from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
df1[categorical_features] = df1[categorical_features].apply(lambda col: le1.fit_transform(col))

df1=pd.concat([xencoded,df1[numerical_features]],axis=1)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features = getmask(df1), sparse=False)

X_test = pd.get_dummies(df1, prefix_sep='_', drop_first=True)
km = KModes(n_clusters=2, init='Huang',max_iter=1000000 ,n_init=5, verbose=1,random_state=0)

clusters1 = km.fit_predict(X_test)

out = pd.DataFrame(data={'custId':df_test['custId'],'Satisfied':clusters1})

out.to_csv('output2.csv',index=False) 