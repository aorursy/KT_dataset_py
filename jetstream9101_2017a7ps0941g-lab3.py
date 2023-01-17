import pandas as pd

import numpy as np

import xgboost as xgb

import os

import seaborn as sns
DATA_DIR = "."

DATA_TRAIN = "/kaggle/input/eval-lab-3-f464/train.csv"

DATA_TEST = "/kaggle/input/eval-lab-3-f464/test.csv"
df_train = pd.read_csv(os.path.join(DATA_DIR, DATA_TRAIN))

df_test = pd.read_csv(os.path.join(DATA_DIR, DATA_TEST))
df_train.head()
df_train.info()
df_train['TotalCharges'].loc[df_train['TotalCharges'] == ' '] = df_train['MonthlyCharges'].loc[df_train['TotalCharges'] == ' ']

df_test['TotalCharges'].loc[df_test['TotalCharges'] == ' '] = df_test['MonthlyCharges'].loc[df_test['TotalCharges'] == ' ']
df_train['TotalCharges'] = pd.to_numeric(df_train['TotalCharges'])

df_test['TotalCharges'] = pd.to_numeric(df_test['TotalCharges'])
numeric_cols = ['custId', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

categorical_cols = ['gender', 'Married', 'Children', 'TVConnection', 'Channel1', 'Channel2', 'Channel3', \

                   'Channel4', 'Channel5', 'Channel6', 'Internet', 'HighSpeed', 'AddedServices', 'Subscription', \

                   'PaymentMethod']
import sklearn

from sklearn.preprocessing import LabelEncoder
data = df_train.copy()

data_eval = df_test.copy()
data_eval.columns
for col in categorical_cols:

    le = LabelEncoder()

    data[col] = le.fit_transform(data[col])

    data_eval[col] = le.transform(data_eval[col])
data.head()
df_train.head()
data.corr()['Satisfied']
print(data[data['Satisfied'] == 1].shape)

print(data[data['Satisfied'] == 0].shape)
sel_cols = ['SeniorCitizen', 'Married', 'Children', 'TVConnection', \

                 'Channel4', 'Channel5', 'Channel6', 'Internet', 'HighSpeed', 'AddedServices', 'Subscription', 'tenure', 'PaymentMethod',\

                 'MonthlyCharges', 'TotalCharges']



data_sel = data[sel_cols]

label = data['Satisfied']

print(data_sel.shape, label.shape)
from sklearn.preprocessing import StandardScaler



norm_features = ['tenure', 'MonthlyCharges', 'TotalCharges']



ss = StandardScaler()

data_sel.loc[:, norm_features] = ss.fit_transform(data_sel[norm_features])
from sklearn.model_selection import train_test_split

x = data_sel.values

y = label.values

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = label.values, test_size = 0.2)
print(x.shape, x_train.shape, x_test.shape, y_train.shape, y_test.shape)
from imblearn.over_sampling import SMOTE, KMeansSMOTE



x_train_os, y_train_os = SMOTE().fit_resample(x_train, y_train)

x_os, y_os = SMOTE().fit_resample(x, y)

# x_train_os, y_train_os = KMeansSMOTE().fit_resample(x_train, y_train)
print(x_train_os.shape, y_train_os.shape, x_os.shape, y_os.shape)
from sklearn.cluster import MiniBatchKMeans
mbkm = MiniBatchKMeans(n_clusters = 2, max_iter = 800)

mbkm.fit(x_os)

x_eval = data_eval[sel_cols]

id_eval = data_eval['custId']
x_eval[norm_features] = ss.transform(x_eval[norm_features])
pred_eval = mbkm.fit_predict(x_eval.values)
out_eval = np.stack([id_eval, pred_eval]).T

out_eval.shape
df_out = pd.DataFrame(pred_eval, index = id_eval, columns = ['Satisfied'])
df_out.to_csv('submission.csv')