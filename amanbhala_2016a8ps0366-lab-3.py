import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

from collections import Counter

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch

from sklearn.preprocessing import StandardScaler, normalize, RobustScaler, MinMaxScaler

from sklearn.decomposition import PCA

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

df = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

df.loc[df['TotalCharges'] == " ", "TotalCharges"] = df["MonthlyCharges"]

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

df.head()
df.info()
missing_count = df.isnull().sum()

missing_count[missing_count > 0]

missing_count
df_dtype_nunique = pd.concat([df.dtypes, df.nunique()],axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
df.isnull().any().any()
df.columns
plt.figure(figsize=(20,20))

sns.heatmap(data=df.corr(),cmap='Blues',annot=True)
trans = SMOTE()

X_values = pd.get_dummies(df.drop(['custId', 'gender','TVConnection','Internet','HighSpeed','SeniorCitizen','Satisfied','Children','Married'], axis=1))

y_values = df['Satisfied']

X_r, y_r = trans.fit_resample(X_values, y_values)

print('Resampled dataset shape %s' % Counter(y_r))
scalerobject = RobustScaler()

X_values = scalerobject.fit_transform(X_values)

df_t = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

df_t.loc[df_t['TotalCharges'] == " ", "TotalCharges"] = df_t["MonthlyCharges"]

df_t['TotalCharges'] = pd.to_numeric(df_t['TotalCharges'])

df_t.head()
X_t = df_t.drop(['custId', 'gender','TVConnection','Internet','HighSpeed','SeniorCitizen','Children','Married'], axis=1)

X_t = pd.get_dummies(X_t)

X_t = scalerobject.fit_transform(X_t)

X_t = normalize(X_t)
#model 1

model1 = Birch(n_clusters=2).fit(X_t)

label1 = model1.labels_

label1

#model 2

model2 = Birch(n_clusters=2,threshold=0.6,branching_factor=50).fit(X_t)

label2 = model2.labels_

label2
out = pd.DataFrame({'Satisfied':label1}, index=df_t['custId'])

out = pd.DataFrame({'Satisfied':label2}, index=df_t['custId'])

out.to_csv("submit34.csv")