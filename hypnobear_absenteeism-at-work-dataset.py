# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
sns.set_style('whitegrid')
df = pd.read_csv('../input/Absenteeism_at_work.csv')
df.head()
print('Shape of dataset is:{}'.format(df.shape))
print('Type of features is:{}'.format(df.dtypes))
df['Absenteeism time in hours'].mean()
sns.jointplot(x='Absenteeism time in hours',y='Seasons',data=df)
sns.jointplot(x='Age',y='Absenteeism time in hours',data=df)
plt.figure(figsize=(12,6))
sns.lmplot(x='Age',y='Absenteeism time in hours',data=df,hue='Day of the week',size=5,aspect=2)
df[df['Day of the week']==6]['Absenteeism time in hours'].mean()
df['Transportation expense'].mean()
sns.jointplot(x='Transportation expense',y='Month of absence',data=df,kind='hex',color='red')
df['Son'].value_counts()
plt.figure(figsize=(10,5))
df[df['Son']!=0]['Absenteeism time in hours'].plot.hist(bins=30)
plt.figure(figsize=(10,6))
df[df['Son']==0]['Absenteeism time in hours'].plot.hist(bins=30)
g = sns.FacetGrid(data=df,col='Son')
g.map(plt.hist,'Absenteeism time in hours')
plt.figure(figsize=(14,6))
df[df['Son']==0]['Age'].plot.hist(bins=30)
plt.figure(figsize=(14,6))
df[df['Son']!=0]['Age'].plot.hist(bins=30)
reason_27 = df['Reason for absence']==27
reasons = df['Reason for absence']
plt.figure(figsize=(10,5))
sns.distplot(df['Reason for absence'])
df[df['Reason for absence']==27].count()
df[df['Absenteeism time in hours']==0].count()
df.head(10)
roa = df.groupby('Reason for absence')
roa['Absenteeism time in hours'].max()
X = df.iloc[:, 1:20].values
y = df.iloc[:,20:].values.reshape(-1,1)
from sklearn.svm import SVR
svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(X,y.ravel())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
prediction = sc_y.inverse_transform(svr_regressor.predict(sc_X.transform(X)))
X.shape
y.shape
y_pred = svr_regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)
from sklearn.metrics import mean_squared_error, explained_variance_score
print('MSE:{}'.format(mean_squared_error(y_test,y_pred)))
print('Explained variance score:{}'.format(explained_variance_score(y_test,y_pred)))
