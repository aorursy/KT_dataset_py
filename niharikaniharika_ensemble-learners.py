#importing all the required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

%matplotlib inline

import seaborn as sns

from sklearn import model_selection

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb

from xgboost import XGBRegressor

from mlxtend.regressor import StackingRegressor
#Loading the dataset

train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")

test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")
train.head()
test.head()
print("Nulls in training dataset:",train.isnull().sum().sum())

print("Nulls in testing dataset:",test.isnull().sum().sum())
print("Shape of the training dataset: ",train.shape)

print("Shape of the testing dataset : ",test.shape)
#Sale Price distribution

plt.figure(figsize=(10,5))

ax=sns.distplot(train["count"], hist=True , color='skyblue')

ax.text(x=0.97, y=0.97,transform=ax.transAxes, s="Skewness: %f" % train["count"].skew(),\

        fontweight='demibold', fontsize=16, verticalalignment='top', horizontalalignment='right',\

         color='teal')

ax.text(x=0.97, y=0.91, transform=ax.transAxes, s="Kurtosis: %f" % train["count"].kurt(),\

        fontweight='demibold', fontsize=16, verticalalignment='top', horizontalalignment='right',\

       color='teal')

plt.ylabel('Frequency')

plt.title('No. of total rentals')

plt.show()
#Sale Price distribution

plt.figure(figsize=(10,5))

ax=sns.distplot(np.log1p(train["count"]), hist=True , color='skyblue')

ax.text(x=0.97, y=0.97,transform=ax.transAxes, s="Skewness: %f" % np.log1p(train["count"]).skew(),\

        fontweight='demibold', fontsize=16, verticalalignment='top', horizontalalignment='right',\

         color='teal')

ax.text(x=0.97, y=0.91, transform=ax.transAxes, s="Kurtosis: %f" % np.log1p(train["count"]).kurt(),\

        fontweight='demibold', fontsize=16, verticalalignment='top', horizontalalignment='right',\

       color='teal')

plt.ylabel('Frequency')

plt.title('No. of total rentals')

plt.show()
#removing the outliars

Q1 =np.log1p(train['count']).quantile(0.25)

Q3 =np.log1p(train['count']).quantile(0.75)

IQR = Q3 - Q1

filter=(np.log1p(train['count']) >= Q1 - 1.5 * IQR) & (np.log1p(train['count'])<= Q3 + 1.5 *IQR)

train=train.loc[filter]

train.shape
#adding the additional columns required

train['hour'] = pd.DatetimeIndex(train['datetime']).hour

train['day'] = pd.DatetimeIndex(train['datetime']).day

train['month'] = pd.DatetimeIndex(train['datetime']).month

train['year'] = pd.DatetimeIndex(train['datetime']).year

train["weekday"]=pd.DatetimeIndex(train['datetime']).weekday

test["weekday"]=pd.DatetimeIndex(test['datetime']).weekday

test['hour'] = pd.DatetimeIndex(test['datetime']).hour

test['day'] = pd.DatetimeIndex(test['datetime']).day

test['month'] = pd.DatetimeIndex(test['datetime']).month

test['year'] = pd.DatetimeIndex(test['datetime']).year

d=test["datetime"]

test["weekend"]=0

test.loc[(train["holiday"]==0) & (test["workingday"]==0),"weekend"]=1

train["weekend"]=0

train.loc[(train["holiday"]==0) & (train["workingday"]==0),"weekend"]=1
#correlation of columns with the count column

corr=train[train.columns[1:]].corr()['count'][:]

corr
#number of rentals according to day time

sns.barplot(y='count', x="hour", data=train, palette="bright")
#peak timings according to weather condition

sns.catplot(x="hour",y="count",col="weather",data=train,kind='bar')
#Counts due to weather conditions

sns.barplot(x="weather", y="count",data=train)
#influence of working day on counts as per timings 

plt.figure(figsize=(15,10))

sns.boxplot(y='count', x="workingday", data=train, palette="colorblind",hue='hour')
#number of rentals over the years

sns.lineplot(x="month", y="count",hue="year", markers=True, dashes=False, data=train)  
#one hot encoding season and weather column

columns=["season","weather"]

for col in columns:

    for i in train.groupby(col).count().index:

        c=col+str(i)

        train[c]=0

        for j in train[col]:

            if (j==i):

                train[c].replace({0:1}, inplace=True)

            else:

                train[c]=0

train=train.drop(columns=["season","weather"],axis=1)

train.head(5)
columns=["season","weather"]

for col in columns:

    for i in test.groupby(col).count().index:

        c=col+str(i)

        test[c]=0

        for j in test[col]:

            if (j==i):

                train[c].replace({0:1}, inplace=True)

            else:

                test[c]=0

test=test.drop(columns=["season","weather"],axis=1)

test.head(5)
#mean encoding the hour column as seems to be an important factor 

columns=["hour"]

for x in columns: 

    mean_encode=train.groupby(x)["count"].mean()

    train.loc[:,x]=train[x].map(mean_encode)

    test.loc[:,x]=test[x].map(mean_encode)

    test[x]=test[x] / test[x].max()

    train[x]=train[x] / train[x].max()
train=train.drop(columns=["registered","casual","atemp","datetime","weekday"],axis=1)

d=test["datetime"]

test=test.drop(columns=["atemp","datetime","weekday"],axis=1)
train.shape
from sklearn import model_selection

kfold = model_selection.KFold(n_splits=10, random_state=100)

from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(criterion = 'mse')

parameters = {"max_depth": [5,10,15,20],

             "min_samples_split": [2,4,6,8],

             "min_samples_leaf": [2,4,6,8,10]}

m1= GridSearchCV(reg, parameters, cv=5, verbose=2,n_jobs=-1)

m1.fit(train.drop(["count"], axis=1), train["count"])
decision_tree = model_selection.cross_val_score(m1,train.drop(["count"], axis=1), train["count"], cv=kfold)
print("Accuracy using decision tree: %.2f%%" % (decision_tree.mean()*100.0))
rf=RandomForestRegressor(criterion = 'mse')

parameters = {"max_depth": [5,10,15,20],

             "min_samples_split": [2,4,6,8],

             "min_samples_leaf": [2,4,6,8,10]}

m2=GridSearchCV(rf, parameters, cv=5, verbose=2,n_jobs=-1)

m2.fit(train.drop(["count"], axis=1), train["count"])
random_forest = model_selection.cross_val_score(m2,train.drop(["count"], axis=1), train["count"], cv=kfold)
print("Accuracy using random forest: %.2f%%" % (random_forest.mean()*100.0))
gb=GradientBoostingRegressor(criterion = 'mse')

parameters = {"max_depth": [5,10,15,20],

             "min_samples_split": [2,4,6,8],

             "min_samples_leaf": [2,4,6,8,10]}

m3=GridSearchCV(rf, parameters, cv=5, verbose=2,n_jobs=-1)

m3.fit(train.drop(["count"], axis=1), train["count"])
gradient_boost= model_selection.cross_val_score(m3,train.drop(["count"], axis=1), train["count"], cv=kfold)
print("Accuracy using gradient boost: %.2f%%" % (gradient_boost.mean()*100.0))
xg = xgb.XGBRegressor(criterion = 'mse')

parameters = {"max_depth": [5,10,15,20],

             "learning rate": [0.1,0.01,0.001,0.9],

             "alpha":[0,1,10],}

m4 = GridSearchCV(xg, parameters, cv=5, verbose=2)

m4.fit(train.drop(["count"], axis=1), train["count"])
extreme_gradient_boost = model_selection.cross_val_score(m4,train.drop(["count"], axis=1), train["count"], cv=kfold)
print("Accuracy using extreme gradient boost: %.2f%%" % (extreme_gradient_boost.mean()*100.0))
st= StackingRegressor(regressors=(m1,m2,m3), 

                               meta_regressor=m4,

                               use_features_in_secondary=True)

st.fit(train.drop(["count"], axis=1), train["count"])
stacked = model_selection.cross_val_score(st,train.drop(["count"], axis=1), train["count"], cv=kfold)
print("Accuracy using stacking: %.2f%%" % (stacked.mean()*100.0))
ans=st.predict(test)

ans[ans<0] = min(train["count"])
ans=np.round(ans)

ans
result = pd.DataFrame(data = {"datetime":d,"count":ans})

result.to_csv("stacked",index=False)