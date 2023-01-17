# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
warnings.filterwarnings('ignore')
dataset=pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding='cp1252')
dataset.head(5)
dataset.drop('Unnamed: 0',axis=1,inplace=True)
dataset.describe()
dataset.isnull().any()
plt.figure(figsize=(25,15))

sns.countplot(dataset['Artist.Name'])

plt.show()
dataset.hist(bins=50,figsize=(20,15))

plt.show()
plt.figure(figsize=(15,10))

sns.lineplot(data=dataset['Danceability'],label='Danceability')

sns.lineplot(data=dataset['Popularity'],label='Popularity')

sns.swarmplot(x=dataset['Danceability'],y=dataset['Popularity'])
plt.figure(figsize=(10,8))

sns.countplot(dataset['Genre'])

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(dataset['Loudness..dB..'])

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(dataset['Danceability'])

plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (20, 10))

sns.heatmap(spotify.corr(), annot = True)

plt.show()
sns.distplot(dataset['Liveness'],kde=True, bins=10,color='orange');

sns.distplot(dataset['Valence.'],kde=True, bins=10,color='blue');
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


X=dataset[['Energy','Danceability','Loudness..dB..','Liveness','Valence.','Length.','Acousticness..','Speechiness.']].values

y=dataset['Popularity'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
regressor = LinearRegression()  

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(regressor.coef_)


print(regressor.intercept_)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

df1 = df.head(25)

print(df1)

df1.plot(kind='bar',figsize=(10,8))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

