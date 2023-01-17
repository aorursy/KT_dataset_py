import pandas as pd

from sklearn.cross_validation import train_test_split

from sklearn import svm 

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

import numpy as np

from sklearn.naive_bayes import GaussianNB
data = pd.read_csv('../input/migration_nz.csv')

data.head(10)
data['Measure'].unique()
data['Measure'].replace("Arrivals",0,inplace=True)

data['Measure'].replace("Departures",1,inplace=True)

data['Measure'].replace("Net",2,inplace=True)
data['Measure'].unique()
data['Country'].unique()
data['CountryID'] = pd.factorize(data.Country)[0]

data['CitID'] = pd.factorize(data.Citizenship)[0]
data['CountryID'].unique()
data.isnull().sum()
data["Value"].fillna(data["Value"].median(),inplace=True)
data.isnull().sum()
data.drop('Country', axis=1, inplace=True)

data.drop('Citizenship', axis=1, inplace=True)
from sklearn.cross_validation import train_test_split

X= data[['CountryID','Measure','Year','CitID']].as_matrix()

Y= data['Value'].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(

  X, Y, test_size=0.3, random_state=9)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=70,max_features = 3,max_depth=5,n_jobs=-1)

rf.fit(X_train ,y_train)

rf.score(X_test, y_test)
X = data[['CountryID','Measure','Year','CitID']]

Y = data['Value']

X_train, X_test, y_train, y_test = train_test_split(

  X, Y, test_size=0.3, random_state=9)
grouped = data.groupby(['Year']).aggregate({'Value' : 'sum'})

grouped.head()


#Growth of migration to New-Zeland by year

grouped.plot(kind='line');plt.axhline(0, color='g')

sns.plt.show()
grouped.plot(kind='bar');plt.axhline(0, color='g')

sns.plt.show()
import seaborn as sns

corr = data.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.show()