# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

data = pd.read_csv("/kaggle/input/top50spotify2019/top50.csv",encoding='ISO-8859-1')
data.head()
data.sort_values(by = ['Popularity'],ascending=False)
plt.figure(figsize = (15,7))

ax = sns.swarmplot(x="Genre", y="Popularity", data = data)

ax = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax = plt.title('Genre')
plt.figure(figsize = (15,7))

ax = sns.countplot(x="Genre", data=data)

ax = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax = plt.title('Genre')
plt.figure(figsize = (15,7))

ax = sns.countplot(x="Artist.Name", data=data)

ax = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax = plt.title('Artist.Name')
data.columns
f, axes = plt.subplots(4, 2, figsize=(20,20))



sns.distplot(data["Beats.Per.Minute"],kde = False, ax=axes[0][0])

sns.distplot(data["Energy"],kde = False, ax=axes[0][1])

sns.distplot(data["Danceability"],kde = False, ax=axes[1][0])

sns.distplot(data["Loudness..dB.."],kde = False, ax=axes[1][1])

sns.distplot(data["Liveness"],kde = False, ax=axes[2][0])

sns.distplot(data["Valence."],kde = False, ax=axes[2][1])

sns.distplot(data["Length."],kde = False, ax=axes[3][0])

sns.distplot(data["Speechiness."],kde = False, ax=axes[3][1])
data.columns
import numpy as np; np.random.seed(0)

import seaborn as sns; sns.set()



plt.figure(figsize=(20,20))

heatmap_data = data[['Beats.Per.Minute','Energy','Danceability','Loudness..dB..','Liveness','Valence.',

                   'Length.','Acousticness..','Speechiness.','Popularity']].corr()

ax = sns.heatmap(heatmap_data,annot=True)
data_ml = data.copy()
data_ml.head()
data_ml['Artist.Name'] = data_ml['Artist.Name'].astype('category').cat.codes

data_ml['Genre'] = data_ml['Genre'].astype('category').cat.codes
data_ml.head()
X = data_ml.iloc[:,2:13].values

y = data_ml.iloc[:,13:].values
X
y
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.svm import SVC
clf = SVC(probability=True,kernel='poly',degree=4,gamma='auto')
clf.fit(X_train,y_train.ravel())
train_result = clf.predict(X_train)

test_result = clf.predict(X_test)
from sklearn.metrics import mean_squared_error



print('Training MSE: ', mean_squared_error(y_train, train_result))

print('Test MSE: ', mean_squared_error(y_test, test_result))
indices_train = np.arange(0,len(y_train),1)

indices_test = np.arange(0,len(y_test),1)
indices_train.shape
fig = plt.figure(figsize = (10,10))

plt.subplot(1, 2, 1)

ax1 = sns.scatterplot(indices_train, train_result.ravel(), label = 'train result')

ax1 = sns.scatterplot(indices_train, y_train.ravel(),label = 'actual data')

plt.title('SVR')

plt.subplot(1, 2, 2)

ax2 = sns.scatterplot(indices_test, test_result.ravel(), label = 'train result')

ax2 = sns.scatterplot(indices_test, y_test.ravel(),label = 'actual data')

plt.title('SVR')
from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=2)
knn_clf = neigh.fit(X_train, y_train.ravel())
train_result_knn = knn_clf.predict(X_train)

test_result_knn = knn_clf.predict(X_test)
from sklearn.metrics import mean_squared_error



print('Training MSE: ', mean_squared_error(y_train, train_result_knn))

print('Test MSE: ', mean_squared_error(y_test, test_result_knn))
fig = plt.figure(figsize = (10,10))

plt.subplot(1, 2, 1)

ax1 = sns.scatterplot(indices_train, train_result_knn.ravel(), label = 'train result')

ax1 = sns.scatterplot(indices_train, y_train.ravel(),label = 'actual data')

plt.title('KNN')

plt.subplot(1, 2, 2)

ax2 = sns.scatterplot(indices_test, test_result_knn.ravel(), label = 'train result')

ax2 = sns.scatterplot(indices_test, y_test.ravel(),label = 'actual data')

plt.title('KNN')
from sklearn.tree import DecisionTreeRegressor



dec_clf = DecisionTreeRegressor(max_depth=4)
dec_tree_clf = dec_clf.fit(X_train, y_train.ravel())
train_result_dec = dec_clf.predict(X_train)

test_result_dec = dec_clf.predict(X_test)
from sklearn.metrics import mean_squared_error



print('Training MSE: ', mean_squared_error(y_train, train_result_dec))

print('Test MSE: ', mean_squared_error(y_test, test_result_dec))
fig = plt.figure(figsize = (10,10))

plt.subplot(1, 2, 1)

ax1 = sns.scatterplot(indices_train, train_result_dec.ravel(), label = 'train result')

ax1 = sns.scatterplot(indices_train, y_train.ravel(),label = 'actual data')

plt.title('DT')

plt.subplot(1, 2, 2)

ax2 = sns.scatterplot(indices_test, test_result_dec.ravel(), label = 'train result')

ax2 = sns.scatterplot(indices_test, y_test.ravel(),label = 'actual data')

plt.title('DT')
from sklearn.ensemble import AdaBoostRegressor
regr = AdaBoostRegressor(random_state=0, n_estimators=100)
boost = regr.fit(X, y)
train_result_boost = boost.predict(X_train)

test_result_boost = boost.predict(X_test)
from sklearn.metrics import mean_squared_error



print('Training MSE: ', mean_squared_error(y_train, train_result_boost))

print('Test MSE: ', mean_squared_error(y_test, test_result_boost))
fig = plt.figure(figsize = (10,10))

plt.subplot(1, 2, 1)

ax1 = sns.scatterplot(indices_train, train_result_boost.ravel(), label = 'train result')

ax1 = sns.scatterplot(indices_train, y_train.ravel(),label = 'actual data')

plt.title('boost')

plt.subplot(1, 2, 2)

ax2 = sns.scatterplot(indices_test, test_result_boost.ravel(), label = 'train result')

ax2 = sns.scatterplot(indices_test, y_test.ravel(),label = 'actual data')

plt.title('boost')