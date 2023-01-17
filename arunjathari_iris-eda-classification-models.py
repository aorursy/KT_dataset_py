# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing
df = pd.read_csv('/kaggle/input/iris/Iris.csv')

df.sample(5)
df.describe()
df.isna().sum()
sns.pairplot(df[df.columns.drop('Id')],hue='Species')

df.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 10))

plt.show()
columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species']

from pandas.plotting import andrews_curves

andrews_curves(df[columns], "Species")

plt.show()
sns.countplot(x=df['Species'])

fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(20,6))

print(df['Species'].value_counts(normalize=True))

sns.distplot(df['SepalLengthCm'],bins=50,ax=ax[0,0])

sns.distplot(df['SepalWidthCm'],bins=50,ax=ax[0,1])

sns.distplot(df['PetalLengthCm'],bins=50,ax=ax[1,0])

sns.distplot(df['PetalWidthCm'],bins=50,ax=ax[1,1])

plt.show()
fig,ax = plt.subplots(1,2,figsize=(20,3))

sns.distplot(df['PetalLengthCm'],bins=50,ax=ax[0])

sns.boxplot(df['PetalLengthCm'],ax=ax[1])

plt.suptitle('PetalLengthCm',fontsize=20)

plt.show()
quantile_transformer = preprocessing.QuantileTransformer(

    output_distribution='normal',n_quantiles=len(df), random_state=0)

X_trans = quantile_transformer.fit_transform(df['PetalLengthCm'].values.reshape((len(df),1)))

fig,ax = plt.subplots(1,2,figsize=(20,3))

plt.suptitle('PetalLengthCm',fontsize=20)

sns.distplot(X_trans,bins=50,ax=ax[0])

sns.boxplot(X_trans,ax=ax[1])

plt.show()
fig,ax = plt.subplots(1,2,figsize=(20,3))

sns.distplot(df['PetalWidthCm'],bins=50,ax=ax[0])

sns.boxplot(df['PetalWidthCm'],ax=ax[1])

plt.suptitle('PetalWidthCm',fontsize=20)

plt.show()
X_trans = quantile_transformer.fit_transform(df['PetalWidthCm'].values.reshape((len(df),1)))

fig,ax = plt.subplots(1,2,figsize=(20,3))

sns.distplot(X_trans,bins=50,ax=ax[0])

sns.boxplot(X_trans,ax=ax[1])

plt.suptitle('PetalWidthCm',fontsize=20)

plt.show()
X = df.drop(['Id','Species'],axis=1)

y = df['Species']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.copy(),y.copy(),test_size=0.3,random_state=0)
quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',n_quantiles=len(X_train), random_state=0)

X_train.loc[:,['PetalWidthCm','PetalLengthCm']]=quantile_transformer.fit_transform(X_train[['PetalWidthCm','PetalLengthCm']].values.reshape((len(X_train),2)))

X_test.loc[:,['PetalWidthCm','PetalLengthCm']]=quantile_transformer.transform(X_test[['PetalWidthCm','PetalLengthCm']].values.reshape((len(X_test),2)))
X_train.describe()
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler



class CustomScaler(BaseEstimator):#, TransformerMixin):

    def __init__(self, columns ):#, copy=True, with_mean=True, with_std=True):

        self.scaler = StandardScaler()#(copy, with_mean, with_std)

        self.columns = columns

        self.mean_ = None

        self.std_ = None

    

    def fit(self, X, y=None):

        self.scaler.fit(X[self.columns], y)

        self.mean_ = np.mean(X[self.columns])

        self.std_ = np.std(X[self.columns])

        return self

    

    def transform(self, X, y=None):#, copy=None):

        init_col_order = X.columns

        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns,index=X.index)

        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]

        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
columns_to_scale = ['SepalLengthCm', 'SepalWidthCm',  'PetalWidthCm', 'PetalLengthCm']



scaler = CustomScaler(columns_to_scale)

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test  = scaler.transform(X_test)
X_train.describe()
from sklearn import metrics
from sklearn import svm

model = svm.SVC()

model.fit(X_train,y_train) 

prediction=model.predict(X_test)

print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,y_test))
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)

prediction=model.predict(X_test)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,y_test))
from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier(n_neighbors=3)

model.fit(X_train,y_train)

prediction=model.predict(X_test)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction,y_test))
error_rate = []

x = range(1,40)

for i in x:

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))

plt.plot(x,error_rate,color='blue', linestyle='dashed', marker='o',

 markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')

plt.show()
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()

model.fit(X_train,y_train) 

prediction=model.predict(X_test) 

print('The accuracy of the Decision Tree is ',metrics.accuracy_score(prediction,y_test))
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=0)

model.fit(X_train,y_train) 

prediction=model.predict(X_test) 

print('The accuracy of the Random Forest is ',metrics.accuracy_score(prediction,y_test))
error_rate = []

x = range(1,50,5)

for i in x:

    knn = RandomForestClassifier(n_estimators=i,criterion='entropy',random_state=0)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))

plt.plot(x,error_rate,color='blue', linestyle='dashed', marker='o',

 markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. no. of estimators')

plt.xlabel('K')

plt.xticks(x)

plt.ylabel('Error Rate')

plt.show()