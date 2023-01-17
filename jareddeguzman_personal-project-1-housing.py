import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
house_filepath = '../input/home-data-for-ml-course/train.csv'

df = pd.read_csv(house_filepath, index_col = 'Id')



df.head()

correlation = df.corr()

plt.figure(figsize=(30,30))

sns.heatmap(correlation, annot=True)

newdf = df.copy()

newdf = newdf[['SalePrice', 'YearBuilt', 'GarageCars', 'YearRemodAdd']]



dfWithOverallQual = df.copy()

dfWithOverallQual = dfWithOverallQual[['OverallQual','SalePrice', 'YearBuilt', 'GarageCars', 'YearRemodAdd']]

manyplots = sns.pairplot(dfWithOverallQual)

manyplots.fig.set_size_inches(25,15)
fig, ax = plt.subplots(1,2, figsize=(16,5))

sns.scatterplot(data=df, x='OverallQual', y='GarageCars', ax=ax[0])

sns.scatterplot(data=df, x='OverallQual', y='YearBuilt', ax=ax[1])

fig.show()
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error





#Initializing Variables for the actual algorithm

model = DecisionTreeClassifier()

X = newdf

y = df['OverallQual']



#Splitting data into testing and training data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model.fit(X_train, y_train)



#Actual Predictions

predictions = model.predict(X_test)

actual_values = df['OverallQual'][0:365]

actual_values
mean_absolute_error(actual_values, predictions)