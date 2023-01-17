import sklearn

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.metrics import roc_auc_score
train_csv = pd.read_csv("../input/eval-lab-3-f464/train.csv")

train_csv = train_csv.drop(columns=["Internet"])

test_csv = pd.read_csv("../input/eval-lab-3-f464/test.csv")

test_csv = test_csv.drop(columns=["Internet"])

Y = train_csv["Satisfied"]
categorical_cols = list(train_csv.select_dtypes(object).columns)

numerical_cols = ['tenure', 'TotalCharges', 'MonthlyCharges']
def custom_encoder(df):

    df = pd.get_dummies(data = df,columns=['TVConnection','PaymentMethod'])

    df['gender'].replace({'Male' : 0, 'Female' : 1},inplace=True)

    df[["Married","Children","AddedServices"]].replace({ 'No' : 0 , 'Yes' : 1}, inplace=True)

    df["HighSpeed"].replace({'No internet' : -1,'No' : 0 ,'Yes' : 1},inplace = True)

    df["Subscription"].replace({'Monthly':0.1,'Biannually':0.5,'Annually':1.0},inplace=True)

    df[["Channel1","Channel2","Channel3","Channel4","Channel5","Channel6"]].replace({'No tv connection' : -1 , 'No' : 0 , 'Yes' : 1 }, inplace=True)

    return df
train_csv["TotalCharges"] = pd.to_numeric(train_csv["TotalCharges"],errors = 'coerce')

train_csv['TotalCharges'].fillna((train_csv['TotalCharges'].mean()), inplace=True)

test_csv["TotalCharges"] = pd.to_numeric(test_csv["TotalCharges"],errors = 'coerce')

test_csv['TotalCharges'].fillna((test_csv['TotalCharges'].mean()), inplace=True)
train_csv = custom_encoder(train_csv)

test_csv = custom_encoder(test_csv)



X = train_csv.drop(columns=["Satisfied", "custId"])

X_test = test_csv.drop(columns=["custId"])



scaler = preprocessing.MinMaxScaler()



X.loc[:,numerical_cols] = scaler.fit_transform(X[numerical_cols])



scaler = preprocessing.MinMaxScaler()

X_test.loc[:,numerical_cols] = scaler.fit_transform(X_test[numerical_cols])

y = train_csv["Satisfied"]
# from sklearn.ensemble import ExtraTreesClassifier

# etc = ExtraTreesClassifier(n_estimators=2000)

# etc.fit(X, y)

# preds = etc.predict(X)

# print(roc_auc_score(y, preds))

# res = sorted(zip(list(test_csv.columns), etc.feature_importances_), key = lambda x: x[1], reverse=True) 

# res = pd.DataFrame(res)

# res.set_index(0).plot.bar()
from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.metrics import roc_auc_score
model = KMeans(n_clusters=2)

preds = model.fit_predict(X,y)
sub_preds = model.predict(X_test)
sub_csv = pd.DataFrame()

sub_csv["custId"] = test_csv["custId"]



##################

sub_preds_ = -(sub_preds-1)

##################



sub_csv["Satisfied"] = sub_preds_

print(sub_csv["Satisfied"].value_counts())