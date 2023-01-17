# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from IPython.core.display import HTML, display
from string import Template
from sklearn.model_selection import train_test_split
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# read in data 
df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data = df.to_dict(orient='records')
# convert strings to numbers
def label_encode(df, col):
    col2idx = {k: i for i, k in enumerate(df[col].unique())}
    df[col] = df[col].map(col2idx)
    return col2idx
df = df.replace('Yes', 1)
df = df.replace('No', 0)
for col in df.columns:
    if 'No internet service' in df[col].unique():
        df[col] = df[col].replace('No internet service', 3)
    if 'No phone service' in df[col].unique():
        df[col] = df[col].replace('No phone service', 3)
gender2idx = label_encode(df, 'gender')
is2idx = label_encode(df, 'InternetService')
contract2idx = label_encode(df, 'Contract')
payment2idx = label_encode(df, 'PaymentMethod')
phoneservice2idx = label_encode(df, 'PhoneService')

df['OnlineSecurity'] = df['OnlineSecurity'].astype(float)
df['InternetService'] = df['InternetService'].astype(float)
df['OnlineBackup'] = df['OnlineBackup'].astype(float)
df['DeviceProtection'] = df['DeviceProtection'].astype(float)
df['TechSupport'] = df['TechSupport'].astype(float)
df['StreamingTV'] = df['StreamingTV'].astype(float)
df['StreamingMovies'] = df['StreamingMovies'].astype(float)
df['TotalCharges'] = df['TotalCharges'].astype(float, errors='ignore')
charges = []
for c in df['TotalCharges'].values:
    try:
        charges.append(float(c))
    except:
        charges.append(0)
df['TotalCharges'] = pd.Series(charges)
df['TotalCharges'] = df['TotalCharges'].replace(0, df['TotalCharges'].median())
# feature engineering
df['Lifetime'] = df['tenure'] * df['MonthlyCharges']
df.describe()
df['Churn'].value_counts()
sns.set()
sns.distplot(df['MonthlyCharges'], kde=False)
sns.set()
sns.distplot(df['tenure'], kde=False)
sns.heatmap(df.corr())
X = df[['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Lifetime']]
# upsample 
oversample = SMOTE()
X, y = oversample.fit_resample(X, df['Churn'])
Counter(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
param = {'max_depth': 5, 'eta': 1, 'objective': 'binary:logistic'}
num_round = 15
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, watchlist)
preds = bst.predict(dtest)
labels = dtest.get_label()
y_pred = [0 if i < 0.5 else 1 for i in preds]

print(classification_report(labels, y_pred))
xgb.plot_importance(bst)
