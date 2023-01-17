# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
missing_values = ["n/a", "na", "--","-"]
newsdat=pd.read_csv("/kaggle/input/online-news-popularity/OnlineNewsPopularity.csv",na_values = missing_values)
newsdat.shape
newsdat.drop_duplicates()
newsdat.isnull().values.any()
newsdat=newsdat.drop(['url'],axis=1)

newsdat.head(10)
newsdat.isnull().sum()
newsdat.describe()
plt.figure(figsize=(40,30))

cor = newsdat.corr(method ='pearson')

sns.heatmap(cor, cmap="RdYlGn")

plt.show()
newsdat1=newsdat.drop([' n_non_stop_words',' n_unique_tokens',' kw_avg_min',' kw_avg_avg',' self_reference_avg_sharess'],axis=1)
y=newsdat1[' shares']

X=newsdat1.drop(' shares',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(min_samples_split=2)

rf.fit(X_train, y_train)
predicted_test = rf.predict(X_test)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predicted_test))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, predicted_test))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predicted_test)))
X_train.shape
def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)

fi = rf_feat_importance(rf, X_train); 
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

plot_fi(fi[:45]);
x = newsdat1.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df = pd.DataFrame(x_scaled)
y=df[54]

X=df.drop(54,axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rf = RandomForestRegressor(min_samples_split=5)

rf.fit(X_train, y_train)
ac=[]

for x in range(2,11):

    rf = RandomForestRegressor(min_samples_split=x)

    rf.fit(X_train, y_train)

    predicted_test1 = rf.predict(X_test)

    ac.append(np.sqrt(metrics.mean_squared_error(y_test, predicted_test1)))
b=[2,3,4,5,6,7,8,9,10]

plt.plot(b,ac)
rf = RandomForestRegressor(min_samples_split=9)

rf.fit(X_train, y_train)

predicted_test = rf.predict(X_test)



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predicted_test))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, predicted_test))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predicted_test)))