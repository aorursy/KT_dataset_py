import pandas as pd 

import numpy as np 

import seaborn as sns 

import matplotlib.pyplot as plt
audi=pd.read_csv('../input/used-car-dataset-ford-and-mercedes/audi.csv')
audi.head()
audi.shape
audi.info()
audi.isnull().sum()
audi.describe()
audi.corr()
plt.figure(figsize=(10,10))

sns.heatmap(audi.corr(), annot=True)
audi.isnull().sum()


fig = plt.figure(figsize=(18,6))

fig.add_subplot(1,2,1)

sns.countplot(audi['transmission'])

fig.add_subplot(1,2,2)

sns.countplot(audi['fuelType'])
sns.pairplot(data=audi, palette="husl")
# this plot shows that the price for some models has less variance than other:

# this might suggest that a specific predict-model for car-model can lead to better overall performance

# (while the generic model can be used to manage those car-model that do not have many rows in dataset

# and so the specific model would lead to poor performance)

sns.catplot(x = 'model', y= 'price', data = audi, kind='point', aspect=4);
# these plots show how much data we can rely on, for any car model

audi.groupby('model').count()['year'].values

plt.figure(figsize=(15, 6))

plt.bar(audi.groupby('model').count()['year'].index,audi.groupby('model').count()['year'].values,color='#005500', alpha=0.7, label='Number or records')

plt.xticks(audi.groupby('model').count()['year'].index, (audi.groupby('model').count()['year'].index), rotation = 90);

plt.xlabel('Car Model')

plt.ylabel('Number of Records')

plt.ylim([0,2500])

plt.suptitle('Number of Records vs Car Model')

plt.legend();
num_cols = audi.select_dtypes(exclude=['object'])



fig = plt.figure(figsize=(20,8))



for col in range(len(num_cols.columns)):

    fig.add_subplot(2,4,col+1)

    sns.distplot(num_cols.iloc[:,col], hist=False, rug=True, kde_kws={'bw':0.1}, label='UV')

    plt.xlabel(num_cols.columns[col])



plt.tight_layout()
num_cols = audi.select_dtypes(exclude=['object'])



fig = plt.figure(figsize=(20,8))



for col in range(len(num_cols.columns)):

    fig.add_subplot(2,4,col+1)

    sns.scatterplot(x=num_cols.iloc[:,col], y=audi['price'], label='MV')

    plt.xlabel(num_cols.columns[col])



plt.tight_layout()

audi2 =pd.get_dummies(audi, columns= ['model', 'transmission', 'fuelType'])
audi2
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
X = audi2.drop(['price'], axis=1)

y = audi2['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7) 
model = LinearRegression()

model.fit(X_train,y_train)

y_predict = model.predict(X_test)

y_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from math import sqrt
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVC

from sklearn.linear_model import Ridge,ElasticNet

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import VotingRegressor

from sklearn.metrics import accuracy_score

from sklearn.linear_model import Lasso
log_clf=LinearRegression()

rnd_clf = RandomForestRegressor()

rid_clf = Ridge(alpha=2,max_iter=1000,random_state=1)

ele_clf = ElasticNet()

gbr_clf=GradientBoostingRegressor()

lss_clf=Lasso(alpha = 500)
voting_clf = VotingRegressor([('lr', log_clf), ('rf', rnd_clf), ('rnd', rnd_clf), ('ele', ele_clf), ('gbr', gbr_clf), ('lss', lss_clf)])

voting_clf.fit(X_train, y_train)
for clf in (log_clf, rnd_clf, rid_clf, voting_clf,ele_clf, gbr_clf,lss_clf):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, 'r2_score', r2_score(y_test, y_pred))
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=7)
voting_clf2 = VotingRegressor([('lr', log_clf), ('rf', rnd_clf), ('rnd', rnd_clf), ('ele', ele_clf), ('gbr', gbr_clf),('lss', lss_clf)])

voting_clf.fit(X_train, y_train)
for clf in (log_clf, rnd_clf, rid_clf, voting_clf2,ele_clf, gbr_clf,lss_clf):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, 'r2_score', r2_score(y_test, y_pred))
# min max scaller didnt add any vlaue to the result 
scaler2 = StandardScaler()

X_scaled2 = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled2, y, test_size=0.2, random_state=7)
voting_clf3 = VotingRegressor([('lr', log_clf), ('rf', rnd_clf), ('rnd', rnd_clf), ('ele', ele_clf), ('gbr', gbr_clf),('lss', lss_clf)])

voting_clf.fit(X_train, y_train)
for clf in (log_clf, rnd_clf, rid_clf, voting_clf3,ele_clf, gbr_clf,lss_clf):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, 'r2_score', r2_score(y_test, y_pred))
import xgboost as xgb
model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1, max_depth = 10, n_estimators = 100)

model.fit(X_train, y_train)
y_predict = model.predict(X_test)
result = model.score(X_test, y_test)



print("Accuracy : {}".format(result))