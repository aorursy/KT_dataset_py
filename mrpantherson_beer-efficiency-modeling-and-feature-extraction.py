# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# magic numbers bad

pal1 = '#ac4b1c'

seed = 121285



df = pd.read_csv('/kaggle/input/beer-efficiency/beer_efficiency.csv')

df.info()
fig, ax = plt.subplots(1,3, figsize=(18,6))



sns.distplot(df['abv'], ax=ax[0], color=pal1, kde=False)

sns.distplot(df['calories'], ax=ax[1], color=pal1, kde=False)

sns.distplot(df['efficiency'], ax=ax[2], color=pal1, kde=False)



ax[0].set_title('ABV')

ax[1].set_title('Calories')

ax[2].set_title('Efficiency')
sns.heatmap(df.corr(), annot=True)
from itertools import chain

from collections import Counter



# split all strings into lists, for example we now have [['Blue', 'Moon', 'Belgian', 'White'], ... ]

words = [x.split() for x in df.name.to_list()]

# now lets just throw all these sub lists into a single list

words = [x for x in chain.from_iterable(words)]



# have counter do the heavy lifting and throw it into a series for easy manipulation

count = pd.Series(Counter(words))

count = count[count > 5]



fig, ax = plt.subplots(figsize=(5,10))

sns.barplot(count, count.index, color=pal1)
from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(max_features=10)

mat = cv.fit_transform(df.name).toarray()

print(cv.vocabulary_)



newdf = pd.DataFrame(mat, columns=cv.vocabulary_.keys())

df = pd.concat([df, newdf], axis=1)

df = df.drop('name', axis=1)
fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(df.corr(), annot=True)
from sklearn.feature_selection import SelectKBest, f_regression



# seperate into input/output features

X = df.drop('efficiency', axis=1)

y = df['efficiency']



ksel = SelectKBest(k='all', score_func=f_regression)

ksel.fit(X, y)



sns.barplot(x=ksel.scores_, y=X.columns, color=pal1)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)



model = LinearRegression()



model.fit(X_train, y_train)

print(model.score(X_train, y_train))

print(model.score(X_test, y_test))
sns.residplot(model.predict(X_test), y_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error



print(mean_squared_error(model.predict(X_test), y_test))

print(mean_absolute_error(model.predict(X_test), y_test))
mape = 100 - (np.mean(np.abs(y_test - model.predict(X_test)) / y_test) * 100)

print(mape)
train_scores = []

test_scores = []



# iterate through all possible features counts, 1 to keep all features

for i in range(1, X.shape[1]):

    ksel = SelectKBest(k=i, score_func=f_regression)

    X_ = ksel.fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_, y, random_state=seed)



    model = LinearRegression()

    model.fit(X_train, y_train)



    train_scores.append(model.score(X_train, y_train))

    test_scores.append(model.score(X_test, y_test))



plt.plot(range(1, X.shape[1]), train_scores)

plt.plot(range(1, X.shape[1]), test_scores)

plt.xlabel('# Features Kept')

plt.ylabel('Score')

plt.legend(['train', 'test'])
from sklearn.model_selection import learning_curve



sizes, t_scores, v_scores = learning_curve(LinearRegression(), X, y, train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0])



t_scores_mean = np.mean(t_scores, axis=1)

v_scores_mean = np.mean(v_scores, axis=1)



fig, ax = plt.subplots()

plt.plot(sizes, t_scores_mean, 'o-', color="r", label="Training score")

plt.plot(sizes, v_scores_mean, 'o-', color="g",label="Test score")

ax.set(xlabel='Training Samples', ylabel='Score')

ax.legend()