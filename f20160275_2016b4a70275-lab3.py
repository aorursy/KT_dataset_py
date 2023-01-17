import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

data.head()
data.isnull().any().any()

data.shape
data.dtypes
def preprocess(data):

    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"],errors = 'coerce')

#     data.dropna(axis = 0,inplace=True)

    data["HighSpeed"].replace({'No internet' : 0,'No' : 1 ,'Yes' : 2},inplace = True)

    data['gender'].replace({'Male' : 0, 'Female' : 1},inplace=True)

    data[["Married","Children","AddedServices"]] = data[["Married","Children","AddedServices"]].replace({ 'No' : 0 , 'Yes' : 1})

    data[["Channel1","Channel2","Channel3","Channel4","Channel5","Channel6"]]=data[["Channel1","Channel2","Channel3","Channel4","Channel5","Channel6"]].replace({'No tv connection' : 0 , 'No' : 1 , 'Yes' : 2 })

    data["Subscription"].replace({'Monthly':1,'Biannually':6,'Annually':12},inplace=True)

    data = pd.get_dummies(data = data,columns=['TVConnection','PaymentMethod'])

    return data
data = preprocess(data)

data.head()
data.dropna(axis = 0,inplace=True)
data.isnull().any().any()
test_data = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

test_data = preprocess(test_data)

test_data.head()
test_data["TotalCharges"].fillna(value = data["TotalCharges"].mean(),inplace=True)
test_data.isnull().any().any()
data.columns.values
cols = ['gender', 'SeniorCitizen', 'Married', 'Children',

       'Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5',

       'Channel6', 'HighSpeed', 'AddedServices',

       'Subscription', 'tenure', 'MonthlyCharges', 'TotalCharges', 'TVConnection_Cable', 'TVConnection_DTH',

       'TVConnection_No', 'PaymentMethod_Bank transfer',

       'PaymentMethod_Cash', 'PaymentMethod_Credit card',

       'PaymentMethod_Net Banking']
X = data[cols]

y = data['Satisfied']

X.head()
X_test = test_data[cols]
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scal_cols = ['TotalCharges','tenure','MonthlyCharges']



X.loc[:,  scal_cols] = scaler.fit_transform(X[scal_cols])
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scal_cols = ['TotalCharges','tenure','MonthlyCharges']



X_test.loc[:,  scal_cols] = scaler.fit_transform(X_test[scal_cols])
X.columns.values
plt.scatter(X['TotalCharges'],X['tenure'],c=y)
# from sklearn.cluster import KMeans

# from sklearn.metrics import roc_auc_score

# from sklearn.metrics import accuracy_score



# kmeans = KMeans(n_clusters=2).fit(X)



# y_pred = kmeans.predict(X)



# acc = roc_auc_score(y,y_pred)

# print(acc)
# y_test = kmeans.predict(X_test)

# len(X_test)
# final  =pd.DataFrame(test_data['custId'])

# final['Satisfied'] = y_test

# final.to_csv('predicted.csv',encoding='utf-8',index=False)
NCLUSTERS=10
from sklearn.cluster import KMeans

from sklearn.metrics import roc_auc_score



kmeans = KMeans(n_clusters=NCLUSTERS, n_init=100, n_jobs=-1).fit(X)



centers = kmeans.cluster_centers_
# plt.scatter(X['TotalCharges'],X['tenure'],c=kmeans.predict(X))
y[kmeans.predict(X)==0].value_counts()[0]
# cluster_preds = [y[kmeans.predict(X)==i].value_counts().sort_values(ascending=False).index[0] for i in range(NCLUSTERS)]



temp = [y[kmeans.predict(X)==i].value_counts()[0] * 3 <= y[kmeans.predict(X)==i].value_counts()[1]  for i in range(NCLUSTERS)]

temp

# cluster_preds = temp.replace({'False':0 , 'True': 1} )
cluster_preds = np.array(temp).astype(int)

cluster_preds
cluster_preds
roc_auc_score([cluster_preds[i] for i in kmeans.predict(X)], y)
y_pred_test = [cluster_preds[i] for i in kmeans.predict(X_test)]
plt.scatter(X['MonthlyCharges'],X['tenure'],c=y)
final  =pd.DataFrame(test_data['custId'])

final['Satisfied'] = y_pred_test

final.to_csv('predicted.csv',encoding='utf-8',index=False)
final['Satisfied'].value_counts()
temp = pd.read_csv('predicted.csv')

temp['Satisfied'].value_counts()