import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm as tq

from sklearn.preprocessing import StandardScaler

from google.colab import drive

from sklearn.cluster import KMeans

from sklearn.metrics import roc_auc_score



drive.mount('/content/drive')

%matplotlib inline
df = pd.read_csv("/content/drive/My Drive/ML_EvalLab/eval-lab-3-f464/train.csv")

df.loc[df.tenure==0,'TotalCharges']='0'
test_df = pd.read_csv("/content/drive/My Drive/ML_EvalLab/eval-lab-3-f464/test.csv")

test_df.loc[test_df.tenure==0,'TotalCharges']='0'
df
df['Married'] = df['Married'].eq('Yes').mul(1)

df['Children'] = df['Children'].eq('Yes').mul(1)

df['Internet'] = df['Internet'].eq('Yes').mul(1)

df['AddedServices'] = df['AddedServices'].eq('Yes').mul(1)

df['gender'] = df['gender'].eq('Male').mul(1)
test_df['Married'] = test_df['Married'].eq('Yes').mul(1)

test_df['Children'] = test_df['Children'].eq('Yes').mul(1)

test_df['Internet'] = test_df['Internet'].eq('Yes').mul(1)

test_df['AddedServices'] = test_df['AddedServices'].eq('Yes').mul(1)

test_df['gender'] = test_df['gender'].eq('Male').mul(1)
one_hot_features =['TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','HighSpeed','Subscription','PaymentMethod']

onehot = pd.get_dummies(data=df, columns = one_hot_features,prefix = one_hot_features)

df = df.merge(onehot)

onehot = pd.get_dummies(data=test_df, columns = one_hot_features,prefix = one_hot_features)

test_df = test_df.merge(onehot)

df = df.drop(one_hot_features,axis=1)

test_df = test_df.drop(one_hot_features,axis=1)
df
scaler = StandardScaler()

df[['tenure','MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure','MonthlyCharges', 'TotalCharges']])

test_df[['tenure','MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(test_df[['tenure','MonthlyCharges', 'TotalCharges']])
df
X = df.loc[:, ~df.columns.isin(['Satisfied','custId'])]

y = df["Satisfied"]

X_test = test_df.loc[:, ~test_df.columns.isin(['custId'])]
X
n_clusters = 167

clf = KMeans(n_clusters = n_clusters, random_state=1234).fit(X)

y_pred_train = clf.predict(X)
dic = {}

ones = [0]*n_clusters

zeros = [0]*n_clusters

 

for i in range(len(X)):

    if y.iloc[i] == 0:

        zeros[y_pred_train[i]] +=1

    else:

        ones[y_pred_train[i]] +=1

 

for i in range(n_clusters):

    if ones[i]>zeros[i]:

        dic[i] = 1

    else:

        dic[i] = 0

y_pred = clf.predict(X_test)

for i in range(len(y_pred)):

    y_pred[i] = dic[y_pred[i]]
acc = roc_auc_score(y,y_pred_train)

print(acc)
submission = pd.concat([test_df.custId,pd.DataFrame(data=y_pred)],axis=1)

submission.columns = ['custId','Satisfied']

submission.to_csv('/content/drive/My Drive/ML_EvalLab/eval-lab-3-f464/submit_1.csv',index=False)