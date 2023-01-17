# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/BlackFriday.csv")
data.head()
data.tail()
data.info()
data.describe()
data.shape
data.hist()
data.corr()
import seaborn as sns 
corr = data.corr() 
sns.heatmap(corr,
            xticklabels=corr.columns.values, yticklabels=corr.columns.values)

corr=data.corr().abs()
n_most_correlated=12
#'SalePrice' ile en yüksek korelasyona sahip özellikler elde edilir.
most_correlated_feature=corr['Purchase'].sort_values(ascending=False)[:n_most_correlated].drop('Purchase')
#En yüksek korelasyona sahip özelliklerin adları elde edilr. 
most_correlated_feature_name=most_correlated_feature.index.values
f, ax = plt.subplots(figsize=(10, 4))
plt.xticks(rotation='90')
sns.barplot(x=most_correlated_feature_name, y=most_correlated_feature)
plt.title("Purchase ile en fazla korelasyona sahip öznitelikler")
corr=data.corr().abs()
n_most_correlated=12
#'SalePrice' ile en yüksek korelasyona sahip özellikler elde edilir.
most_correlated_feature=corr['Product_Category_1'].sort_values(ascending=False)[:n_most_correlated].drop('Product_Category_1')
#En yüksek korelasyona sahip özelliklerin adları elde edilr. 
most_correlated_feature_name=most_correlated_feature.index.values
f, ax = plt.subplots(figsize=(10, 4))
plt.xticks(rotation='90')
sns.barplot(x=most_correlated_feature_name, y=most_correlated_feature)
plt.title("Product_Category_1 ile en fazla korelasyona sahip öznitelikler")
data.isnull().sum()
data.isnull().sum().sum()
def eksik_deger_tablosu(df): 
    mis_val = data.isnull().sum()
    mis_val_percent = 100 * data.isnull().sum()/len(data)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return mis_val_table_ren_columns
eksik_deger_tablosu(data)
data['Product_Category_2'] = data['Product_Category_2'].fillna('20')
data['Product_Category_3'] = data['Product_Category_3'].fillna('50')
data
import seaborn as sns
sns.boxplot(x=data['Purchase'])

data['TotalProduct'] = data['Product_Category_1'] + str(int(data['Product_Category_1']))
data = data.append(data.iloc[:6])
data.tail(6)

from sklearn import preprocessing
x = data[['Purchase']].values.astype(int)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data['Purchase2'] = pd.DataFrame(x_scaled)

data

#Eğitim için ilgili öznitlelik değerlerini seç
data = pd.concat([data, pd.get_dummies(data.Product_Category_1 )], axis=1)
data = data.drop(['Product_Category_1'],axis=1)
data = pd.concat([data, pd.get_dummies(data.Product_ID )], axis=1)
data = data.drop(['Product_ID'],axis=1)
data = pd.concat([data, pd.get_dummies(data.Gender)], axis=1)
data = data.drop(['Gender'],axis=1)
data = pd.concat([data, pd.get_dummies(data.Product_Category_1 )], axis=1)
data = data.drop(['Product_Category_1'],axis=1)
data = pd.concat([data, pd.get_dummies(data.Product_Category_1 )], axis=1)
data = data.drop(['Product_Category_1'],axis=1)


X = data.iloc[:, :-1].values

#Sınıflandırma öznitelik değerlerini seç
Y = data.iloc[:, -1].values
Y
from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = knn , X = x_train, y=y_train,cv=10)



