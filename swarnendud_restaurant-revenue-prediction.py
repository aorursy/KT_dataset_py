# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/restaurant-revenue-prediction/train.csv")
df_train.info()
df_test = pd.read_csv("../input/restaurant-revenue-prediction/test.csv")
df_train.describe()
sns.distplot(df_train['revenue'])
df_train['City'].value_counts()
df_train['City Group'].value_counts()
df_test['City Group'].value_counts()
df_train['Type'].value_counts()
df_test['Type'].value_counts()
sns.catplot(x="Type", y="revenue", kind="swarm", data=df_train);
sns.catplot(x="City Group", y="revenue", kind="swarm", data=df_train);
df_train['Open Date'].value_counts()
plt.figure(figsize=(20,20))

sns.heatmap(df_train.corr())
sns.scatterplot(x="revenue", y="P2", hue="Type", data=df_train)
Y_train = df_train['revenue']
Y_train.head()
df_feat = df_train.drop(['revenue'], axis=1)

df_feat = df_feat.drop(['Id'], axis=1)

df_feat = df_feat.drop(['Open Date'], axis=1)

df_feat = df_feat.drop(['City'], axis=1)

df_feat = df_feat.drop(['Type'], axis=1)
df_feat.head()
df_test = df_test.drop(['Id'], axis=1)

df_test = df_test.drop(['Open Date'], axis=1)

df_test = df_test.drop(['City'], axis=1)

df_test = df_test.drop(['Type'], axis=1)
df_test.head()
total = df_feat.isnull().sum().sort_values(ascending=False)

percent = (df_feat.isnull().sum()/df_feat.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
df_feat.columns




df_pcols = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15',

       'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35',

       'P36', 'P37']

for i, column in enumerate(df_pcols):

    plt.figure()

    sns.distplot(df_feat[column])
from sklearn.preprocessing import StandardScaler
df_feat = pd.get_dummies(df_feat, prefix=['CityGroup'])
df_test = pd.get_dummies(df_test, prefix=['CityGroup'])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_features_train = scaler.fit_transform(df_feat)
scaled_features_test = scaler.fit_transform(df_test)
scaled_features_df_feat = pd.DataFrame(scaled_features_train, index=df_feat.index, columns=df_feat.columns)
scaled_features_df_test = pd.DataFrame(scaled_features_test, index=df_test.index, columns=df_test.columns)
scaled_features_df_feat.head()
scaled_features_df_test.head()
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators=20)

regr.fit(scaled_features_df_feat,Y_train)
predictions = regr.predict(scaled_features_df_test)
from sklearn.linear_model import Lasso
regr2 = Lasso(alpha=0.1)

regr2.fit(scaled_features_df_feat,Y_train)
predictions2 = regr2.predict(scaled_features_df_test)
df_test1 = pd.read_csv("../input/restaurant-revenue-prediction/test.csv")

sub = pd.DataFrame()

sub['Id'] = df_test1['Id']

sub['Prediction'] = predictions

sub.to_csv('submission.csv',index=False)
df_test1 = pd.read_csv("../input/restaurant-revenue-prediction/test.csv")

sub = pd.DataFrame()

sub['Id'] = df_test1['Id']

sub['Prediction'] = predictions2

sub.to_csv('submission2.csv',index=False)
from IPython.display import FileLink

FileLink(r'submission.csv')
from IPython.display import FileLink

FileLink(r'submission.csv')