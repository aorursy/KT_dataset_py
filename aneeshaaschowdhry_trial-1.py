# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import TimeSeriesSplit



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/tv-power-consumption-data/Q3-data.csv')

df.set_index('Time',inplace=True)

print(df.shape)
df.head()
fig = px.line(df, x=df.index.values, y="TV", color='House')

fig.show()
def denoise(data,signal, name):

  clf = KNeighborsRegressor(n_neighbors=60, weights='uniform')

  for i in data.House.unique():

    clf.fit(data[data.House==i].index.values[:, np.newaxis], data[data.House==i].loc[:, signal])

    data.loc[data.House==i, name] = clf.predict(data[data.House==i].index.values[:, np.newaxis])
denoise(df,'TV','denoisedtv')
fig = px.line(df, x=df.index.values, y="denoisedtv", color='House')



fig.show()
scaler = StandardScaler()



for  i,house in enumerate(df.House.unique()):

  if i==0:

    Xscaled = scaler.fit_transform(df[df.House==house][['denoisedtv']])

  else:

    Xscaled = np.append(Xscaled,scaler.fit_transform(df[df.House==i][['denoisedtv']]))
from hmmlearn import hmm
for i in df.House.unique():

  X = df[df['House']==i][['denoisedtv']].values

  model = hmm.GaussianHMM(n_components=2, covariance_type="full", random_state=7).fit(X)

  df.loc[df['House'] == i, 'Power'] = model.predict(X)
h1 = df[df.House==1]

fig = px.line(h1, x=h1.index.values, y="denoisedtv", color='House')

fig.add_scatter(x=h1.index.values, y=h1['Power'])

fig.show()
h2 = df[df.House==2]

fig = px.line(h2, x=h2.index.values, y="denoisedtv", color='House')

fig.add_scatter(x=h2.index.values, y=h2['Power'])

fig.show()
h3 = df[df.House==3]

fig = px.line(h3, x=h3.index.values, y="denoisedtv", color='House')

fig.add_scatter(x=h3.index.values, y=h3['Power'])

fig.show()
denoise(df,'Agg','denoisedagg')
fig = px.line(df, x=df.index, y="denoisedagg", color='House')

fig.show()
h1 = df[df.House==1]

fig = px.line(h1, x=h1.index.values, y="denoisedtv")

fig.add_scatter(x=h1.index.values, y=h1['denoisedagg'],mode='lines')

fig.add_scatter(x=h1.index.values, y=h1['Power'],mode='lines')

fig.show()
h2 = df[df.House==2]

fig = px.line(h2, x=h2.index.values, y="denoisedtv")

fig.add_scatter(x=h2.index.values, y=h2['denoisedagg'],mode='lines')

fig.add_scatter(x=h2.index.values, y=h2['Power'],mode='lines')

fig.show()
h3 = df[df.House==3]

fig = px.line(h3, x=h3.index.values, y="denoisedtv")

fig.add_scatter(x=h3.index.values, y=h3['denoisedagg'],mode='lines')

fig.add_scatter(x=h3.index.values, y=h3['Power'],mode='lines')

fig.show()
df.groupby(['House','Power']).count()['TV'].unstack().plot(kind='bar')
from sklearn.ensemble import GradientBoostingClassifier
temp_df = df.copy()

def createfeatures(X):

  for i in [5,15,60,100,120]: 

    

    X["rolling_mean_" + str(i)] = X.denoisedagg.rolling(window=i).mean()

    X["rolling_std_" + str(i)] = X.denoisedagg.rolling(window=i).std() 

  return X
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(temp_df[['denoisedagg']], temp_df[['Power']], test_size=0.33, random_state=42)

X_train = createfeatures(X_train)



X_train.bfill(inplace=True)

gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.01, max_depth=5, random_state=11)

gb_clf.fit(X_train, Y_train)

    

print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, Y_train)))

X_test = createfeatures(X_test)

X_test.bfill(inplace=True)

print("Accuracy score (Test set): {0:.3f}".format(gb_clf.score(X_test, Y_test)))
gb_clf.feature_importances_
importances = gb_clf.feature_importances_



(pd.Series(importances, index=X_train.columns)

   .nlargest(20)

   .plot(kind='barh')) 
from sklearn.metrics import confusion_matrix

import sklearn.metrics

#from sklearn.metrics import precision_recall_curve

#from sklearn.metrics import plot_precision_recall_curve

import matplotlib.pyplot as plt

y_pred = gb_clf.predict(X_test)

confusion_matrix(Y_test, y_pred)
sklearn.metrics.f1_score(Y_test, y_pred)
sklearn.metrics.recall_score(Y_test, y_pred)
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_precision_recall_curve

import matplotlib.pyplot as plt





from sklearn.metrics import average_precision_score

average_precision = average_precision_score(Y_test, y_pred)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))





disp = plot_precision_recall_curve(gb_clf, X_test, Y_test)

disp.ax_.set_title('2-class Precision-Recall curve: '

                   'AP={0:0.2f}'.format(average_precision))
def createmorefeatures(X):

  X["rolling_q25"] = X.denoisedagg.rolling(window=5).quantile(0.25)

  X["rolling_q75"] = X.denoisedagg.rolling(window=5).quantile(0.75)

  X["rolling_q50"] = X.denoisedagg.rolling(window=5).quantile(0.5)

  X["rolling_iqr"] = X.rolling_q75 - X.rolling_q25

  X["rolling_min"] = X.denoisedagg.rolling(window=5).min()

  X["rolling_max"] = X.denoisedagg.rolling(window=5).max()

  X["rolling_skew"] = X.denoisedagg.rolling(window=5).skew()

  X["rolling_kurt"] = X.denoisedagg.rolling(window=5).kurt()

  return X
X_train = createmorefeatures(X_train)

X_train.bfill(inplace=True)

gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.01, max_depth=5, random_state=11)

gb_clf.fit(X_train, Y_train)

    

print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, Y_train)))

X_test = createmorefeatures(X_test)

X_test.bfill(inplace=True)

print("Accuracy score (Test set): {0:.3f}".format(gb_clf.score(X_test, Y_test)))
