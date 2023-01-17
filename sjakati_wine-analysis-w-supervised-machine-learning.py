# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
init_data = pd.read_csv("../input/winemag-data_first150k.csv")

print("Length of dataframe before duplicates are removed:", len(init_data))

init_data.head()
parsed_data = init_data[init_data.duplicated('description', keep=False)]

print("Length of dataframe after duplicates are removed:", len(parsed_data))

parsed_data.head()
price_variety_df = parsed_data[['price','variety']]

price_variety_df.info()
fig, ax = plt.subplots(figsize=(30,10))

sns.boxplot(x='variety', y='price', data=price_variety_df)

plt.xticks(rotation = 90)

plt.show()
price_variety_clipped = price_variety_df[price_variety_df['price'] > 100]

price_variety_clipped.shape
fig2, ax2 = plt.subplots(figsize=(30,10))

sns.boxplot(x='variety', y='price', data=price_variety_clipped)

plt.xticks(rotation = 90)

plt.show()
## find counts of each type of wine

variety_counts = price_variety_clipped['variety'].value_counts().to_frame()

## standard deviations

variety_std = price_variety_clipped.groupby('variety').std()

variety_counts.index.name = 'variety'

## merge the dataframes

variety_counts = variety_counts.sort_index()

variety_std = variety_std.join(variety_counts)

## show a plot detailing it

regression_plot = sns.regplot(x='variety', y='price', data=variety_std, color="#4CB391")

regression_plot.set(xlabel='Count', ylabel='Standard Deviation')

plt.show()
parsed_data_clipped = parsed_data[parsed_data['price'] > 100]

parsed_data_clipped.shape
## Determine if there is a correlation between the quantity of the points and the price

sns.pairplot(data=parsed_data_clipped[['points', 'price']])
sns.jointplot(x='points', y='price', data=parsed_data_clipped, kind="hex", color="#4CB391")
fig3, ax3 = plt.subplots(figsize=(30,10))

sns.countplot(x='variety', data=price_variety_clipped)

plt.xticks(rotation = 90)

plt.show()
## find the mean price of wine from each country

mean_country_price = parsed_data_clipped.groupby('country')['price'].mean().to_frame()

country_counts = parsed_data_clipped['country'].value_counts().to_frame()

mean_count_df = country_counts.join(mean_country_price)

mean_count_plot = sns.regplot(x='country', y='price', data=mean_count_df, color="#2F32B7")

mean_count_plot.set(xlabel='Count', ylabel='Mean Price')

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
## create the x and y columns

x_unsplit = parsed_data[['country', 'points', 'price', 'province', 'region_1']]

y_unsplit = parsed_data['variety']

## one hot encode the data

x_unsplit = pd.get_dummies(x_unsplit, columns=['country', 'province', 'region_1'])

y_unsplit = pd.get_dummies(y_unsplit, columns=['variety'])

print(x_unsplit.shape, y_unsplit.shape)

## get the training and testing data

X_train, X_test, y_train, y_test = train_test_split(x_unsplit, y_unsplit, random_state=1, train_size=0.65)
X_train_35 = X_train.fillna({"price": 35.})

X_test_35 = X_test.fillna({"price":35.})

clf.fit(X_train_35, y_train)
y_predictions = clf.predict(X_test_35)
from sklearn.metrics import accuracy_score

dt_acc = accuracy_score(y_test,y_predictions)

print(dt_acc)
parsed_data_mean = parsed_data

parsed_data_mean['price'] = parsed_data_mean.groupby('variety').transform(lambda x: x.fillna(x.mean()))

## create the x and y columns

x_unsplit_mean = parsed_data_mean[['country', 'points', 'price', 'province', 'region_1']]

y_unsplit_mean = parsed_data_mean['variety']

## one hot encode the data

x_unsplit_mean = pd.get_dummies(x_unsplit_mean, columns=['country', 'province', 'region_1'])

y_unsplit = pd.get_dummies(y_unsplit_mean, columns=['variety'])

print(x_unsplit_mean.shape, y_unsplit_mean.shape)

## get the training and testing data

X_train_mean, X_test_mean, y_train_mean, y_test_mean = train_test_split(x_unsplit_mean, y_unsplit_mean, random_state=1, train_size=0.65)
clf.fit(X_train_mean, y_train_mean)
y_predictions_mean = clf.predict(X_test_mean)
dt_acc_mean = accuracy_score(y_test_mean,y_predictions_mean)

print(dt_acc_mean)
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier()
forest_clf.fit(X_train_35, y_train)

forest_predictions = forest_clf.predict(X_test_35)

rf_acc = accuracy_score(y_test,forest_predictions)
print(rf_acc)