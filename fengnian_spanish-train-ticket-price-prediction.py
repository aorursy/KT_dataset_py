import numpy as np 

import pandas as pd 

import seaborn as sns

import datetime

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn import linear_model, svm, gaussian_process

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
data = pd.read_csv('../input/renfe.csv', index_col=0)

data.head()
data["price"].describe()
data.columns.values
data.info()
data.shape
data.isnull().sum()/data.shape[0] * 100
data['price'].fillna(data['price'].mean(),inplace=True)
data['train_class'].value_counts()
data['train_class'].fillna("Turista",inplace=True)
data['fare'].value_counts()
data['fare'].fillna("Promo",inplace=True)
data.isnull().sum()
data['price'].describe()
sns.distplot(data['price'])


# train_type

var = 'train_type'

data_train_type = pd.concat([data['price'], data[var]], axis=1)

plt.subplots(figsize=(15,6))

sns.boxplot(x=var, y="price",data=data_train_type)
# train_class

var = 'train_class'

data_train_class = pd.concat([data['price'], data[var]], axis=1)

plt.subplots(figsize=(15,6))

sns.boxplot(x=var, y="price",data=data_train_class)
# origin

var = 'origin'

data_origin = pd.concat([data['price'], data[var]], axis=1)

plt.subplots(figsize=(15,6))

sns.boxplot(x=var, y="price",data=data_origin)
#  destination

var = 'destination'

data_dest = pd.concat([data['price'], data[var]], axis=1)

plt.subplots(figsize=(15,6))

sns.boxplot(x=var, y="price",data=data_dest)
data.drop('insert_date',axis=1,inplace=True)
#strp time

#data['start_date'] = pd.to_datetime(data['start_date'])

#data['end_date'] = pd.to_datetime(data['end_date'])
def dataInterval(start,end):

    start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')

    end = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')

    delta = end - start

    return delta.seconds/3600
data['data_interval'] = data.apply(lambda x:dataInterval(x['start_date'],x['end_date']),axis = 1)
data.drop(['start_date','end_date'],axis=1,inplace=True)
data.head()
f_names = ["origin","destination","train_type","train_class","fare"]

for x in f_names:

    label = preprocessing.LabelEncoder()

    data[x] = label.fit_transform(data[x])

data.head()
corrmat = data.corr()

f, ax = plt.subplots(figsize=(20, 9))

sns.heatmap(corrmat, vmax=0.8, square=True)
# Get the date 

cols = ['origin','destination', 'train_type', 'train_class', 'fare', 'data_interval']

x = data[cols].values

y = data['price'].values

x_scaled = preprocessing.StandardScaler().fit_transform(x)

y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))

X_train,X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.1, random_state=10)
clfs ={

    "LGBMRegressor":LGBMRegressor(),

    "RandomForestRegressir":RandomForestRegressor(n_estimators=200),

    'BayesianRidge':linear_model.BayesianRidge()

}

for clf in clfs:

    try:

        clfs[clf].fit(X_train, y_train)

        #y_pred = clfs[clf].predict(X_test)

        #print(clf + " cost:" + str(np.sum(y_pred-y_test)/len(y_pred)) )

        print(clf+" score:"+ str(clfs[clf].score(X_test,y_test)))

    except Exception as e:

        print(clf + " Error:")

        print(str(e))