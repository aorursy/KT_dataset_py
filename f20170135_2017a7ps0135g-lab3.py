import pandas as pd

import numpy as np

import xgboost as xgb

import os

import seaborn as sns
DATA_DIR = "/kaggle/input/eval-lab-3-f464/"

DATA_TRAIN = "train.csv"

DATA_TEST = "test.csv"
df_train = pd.read_csv(os.path.join(DATA_DIR, DATA_TRAIN))

df_test = pd.read_csv(os.path.join(DATA_DIR, DATA_TEST))
df_train.head()
df_train.info()
df_train.loc[df_train['TotalCharges'] == ' ']['TotalCharges']
df_train[df_train['TotalCharges'] == ' ']['MonthlyCharges'].apply(lambda x: str(x))
df_train['TotalCharges'].loc[df_train['TotalCharges'] == ' '] = df_train['MonthlyCharges'].loc[df_train['TotalCharges'] == ' ']

df_test['TotalCharges'].loc[df_test['TotalCharges'] == ' '] = df_test['MonthlyCharges'].loc[df_test['TotalCharges'] == ' ']
df_train.loc[df_train['TotalCharges'] == ' ']
df_test.loc[df_test['TotalCharges'] == ' ']
df_train.isnull().sum()
df_train.dtypes
df_train['TotalCharges'] = pd.to_numeric(df_train['TotalCharges'])

df_test['TotalCharges'] = pd.to_numeric(df_test['TotalCharges'])
df_train['gender'].unique()
numeric_cols = ['custId', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

categorical_cols = ['gender', 'Married', 'Children', 'TVConnection', 'Channel1', 'Channel2', 'Channel3', \

                   'Channel4', 'Channel5', 'Channel6', 'Internet', 'HighSpeed', 'AddedServices', 'Subscription', \

                   'PaymentMethod']
import sklearn

from sklearn.preprocessing import LabelEncoder
data = df_train.copy()

data_eval = df_test.copy()



# print(data.head())

# print(df_train.head())
data_eval.columns
for col in categorical_cols:

    le = LabelEncoder()

    data[col] = le.fit_transform(data[col])

    data_eval[col] = le.transform(data_eval[col])
data.head()
df_train.head()
data.corr()['Satisfied']
data_dummies = pd.get_dummies(df_train)

print(data_dummies.shape)

data_dummies.head()
data.corr()
data.corr()['Satisfied']
print(data[data['Satisfied'] == 1].shape)

print(data[data['Satisfied'] == 0].shape)
sel_cols = ['SeniorCitizen', 'Married', 'Children', 'TVConnection', \

                 'Channel4', 'Channel5', 'Channel6', 'Internet', 'HighSpeed', 'AddedServices', 'Subscription', 'tenure', 'PaymentMethod',\

                 'MonthlyCharges', 'TotalCharges']



data_sel = data[sel_cols]

label = data['Satisfied']

print(data_sel.shape, label.shape)
df_train['tenure'].unique().shape
data_sel['tenure'].unique()
from sklearn.preprocessing import StandardScaler



norm_features = ['tenure', 'MonthlyCharges', 'TotalCharges']



ss = StandardScaler()

data_sel.loc[:, norm_features] = ss.fit_transform(data_sel[norm_features])
data_sel.head()
from sklearn.model_selection import train_test_split

x = data_sel.values

y = label.values

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = label.values, test_size = 0.2)
print(x.shape, x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(x[:5])
from imblearn.over_sampling import SMOTE, KMeansSMOTE



x_train_os, y_train_os = SMOTE().fit_resample(x_train, y_train)

x_os, y_os = SMOTE().fit_resample(x, y)

# x_train_os, y_train_os = KMeansSMOTE().fit_resample(x_train, y_train)
print(x_train_os.shape, y_train_os.shape, x_os.shape, y_os.shape)
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch, SpectralClustering, AffinityPropagation



km = KMeans(n_clusters = 2, )

# km.fit(x_train)

km.fit(x_os)
b = Birch(n_clusters = 2)

b.fit(x_os)
mbkm = MiniBatchKMeans(n_clusters = 2, max_iter = 300)

mbkm.fit(x_os)
s = SpectralClustering(n_clusters = 2, affinity = 'nearest_neighbors')

# s.fit(x_train)
pred_km = km.predict(x_test)

print(pred_km.shape)
pred_b = b.predict(x_test)

pred_mbkm = mbkm.predict(x_test)
pred_s = s.fit_predict(x_test)
(pred_km == 1-y_test).sum()/pred_km.shape
(pred_b == 1-y_test).sum()/pred_b.shape
(pred_mbkm == 1-y_test).sum()/pred_mbkm.shape
(pred_s == 1-y_test).sum()/pred_s.shape
data_eval.head()
x_eval = data_eval[sel_cols]

id_eval = data_eval['custId']

print(x_eval.shape)
x_eval[norm_features] = ss.transform(x_eval[norm_features])
x_eval.head()
pred_eval = mbkm.fit_predict(x_eval.values)

print(pred_eval.shape)
out_eval = np.stack([id_eval, pred_eval]).T

out_eval.shape
df_out = pd.DataFrame(pred_eval, index = id_eval, columns = ['Satisfied'])
df_out.head(10)
df_out.to_csv('sub11.csv')