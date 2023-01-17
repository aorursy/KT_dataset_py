import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
df = pd.concat(map(pd.read_csv, ['/kaggle/input/used-car-dataset-ford-and-mercedes/audi.csv', '/kaggle/input/used-car-dataset-ford-and-mercedes/bmw.csv','/kaggle/input/used-car-dataset-ford-and-mercedes/hyundi.csv', '/kaggle/input/used-car-dataset-ford-and-mercedes/merc.csv', '/kaggle/input/used-car-dataset-ford-and-mercedes/skoda.csv','/kaggle/input/used-car-dataset-ford-and-mercedes/toyota.csv','/kaggle/input/used-car-dataset-ford-and-mercedes/vauxhall.csv','/kaggle/input/used-car-dataset-ford-and-mercedes/ford.csv','/kaggle/input/used-car-dataset-ford-and-mercedes/vw.csv']))

df
df.info()
sns.heatmap(df.isnull())
df.tax=df.tax.fillna(df[['tax(£)']].max(1))
sns.heatmap(df.isnull())
df.drop(['tax(£)'], axis=1,inplace=True)
df.head()
df.isnull().sum()
df.shape
df.info()
df.corr()
sns.heatmap(df.corr(),annot=True)
fig = plt.figure(figsize=(18,6))
fig.add_subplot(1,2,1)
sns.countplot(df['transmission'])
fig.add_subplot(1,2,2)
sns.countplot(df['fuelType'])
sns.catplot(x = 'year', y= 'price', data = df, kind='point', aspect=4);
sns.relplot(x="price", y="transmission", 
            data=df);
sns.relplot(x="year", y="price", 
            data=df);
num_cols = df.select_dtypes(exclude=['object'])

fig = plt.figure(figsize=(20,8))

for col in range(len(num_cols.columns)):
    fig.add_subplot(2,4,col+1)
    sns.distplot(num_cols.iloc[:,col], hist=False, rug=True, kde_kws={'bw':0.1}, label='UV')
    plt.xlabel(num_cols.columns[col])

plt.tight_layout()
fig = plt.figure(figsize=(20,8))

for col in range(len(num_cols.columns)):
    fig.add_subplot(2,4,col+1)
    sns.scatterplot(x=num_cols.iloc[:,col], y=df['price'], label='MV')
    plt.xlabel(num_cols.columns[col])

plt.tight_layout()
cars
X = cars.drop(['price'], axis=1)
y = cars['price']
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVC
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) 
log_clf=LinearRegression()
bye_clf=BayesianRidge()
rnd_clf = RandomForestRegressor()
rid_clf = Ridge(alpha=2,max_iter=1000,random_state=1)
ele_clf = ElasticNet()
gbr_clf=GradientBoostingRegressor()
lss_clf=Lasso()
voting_clf = VotingRegressor([('lr', log_clf),('bye', bye_clf), ('rf', rnd_clf), ('rnd', rnd_clf), ('ele', ele_clf), ('gbr', gbr_clf), ('lss', lss_clf)])
voting_clf.fit(X_train, y_train)
for clf in (log_clf, rnd_clf, rid_clf, bye_clf, voting_clf,ele_clf, gbr_clf,lss_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, 'r2_score', r2_score(y_test, y_pred))
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=7)
voting_clf2 = VotingRegressor([('lr', log_clf),('bye', bye_clf), ('rf', rnd_clf), ('rnd', rnd_clf), ('ele', ele_clf), ('gbr', gbr_clf),('lss', lss_clf)])
voting_clf.fit(X_train, y_train)
for clf in (log_clf, rnd_clf, rid_clf, bye_clf,voting_clf2,ele_clf, gbr_clf,lss_clf):
    clf.fit(X_train, y_train)
    y_pred2 = clf.predict(X_test)
    print(clf.__class__.__name__, 'r2_score', r2_score(y_test, y_pred2))
scaler2 = StandardScaler()
X_scaled2 = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled2, y, test_size=0.15, random_state=7)
voting_clf3 = VotingRegressor([('lr', log_clf), ('rf', rnd_clf),('bye',bye_clf), ('rnd', rnd_clf), ('ele', ele_clf), ('gbr', gbr_clf),('lss', lss_clf)])
voting_clf.fit(X_train, y_train)
for clf in (log_clf, rnd_clf, rid_clf, voting_clf2,ele_clf, gbr_clf,lss_clf):
    clf.fit(X_train, y_train)
    y_pred3 = clf.predict(X_test)
    print(clf.__class__.__name__, 'r2_score', r2_score(y_test, y_pred3))
