# Importing libraries

import pandas as pd

import numpy as np

import math

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

%matplotlib inline
# We are reading our data

df  = pd.read_csv('../input/usa-housing/USA_Housing.csv')
# Printing information about dataset

df.info()
# Printing the sample of dataset

df.head()
# Describing the dataset 

df.describe()
# Finding if there is any null/missing values in the datasets or not. It's important to remove or

# replace all the missing values before moving further. 

df.isnull().sum()
# correlation plot.

df.corr()
# heatmap

plt.figure(figsize=(12,10))

sns.heatmap(df.corr(), annot=True)
# Price metrics in tabular format

df.corr().Price.sort_values(ascending=False)
# Pair Plot

sns.pairplot(df)
# Scatter Plot 

plt.scatter(df.Price, df[['Avg. Area Income']])
# Displot

sns.distplot(df.Price)
df = df.drop(['Address'], axis=1)

df.head()
from sklearn import preprocessing

pre_process = preprocessing.StandardScaler()
feature = df.drop(['Price'], axis = 1)

label = df.Price



# Now, we have feature and label for machine learning algorithms. Now, we can scale the data by using standard scaler.



feature = pre_process.fit_transform(feature)
#this is how the scaled data looks like.

from sklearn.model_selection import train_test_split

feature_train, feature_test, label_train, label_test = train_test_split(feature, label.values, test_size = 0.2, random_state = 19)
from sklearn import linear_model

linear_regression = linear_model.LinearRegression()

linear_regression.fit(feature_train, label_train)
from sklearn.metrics import r2_score, mean_squared_error



score = r2_score(linear_regression.predict(feature_train), label_train)

error = mean_squared_error(linear_regression.predict(feature_train), label_train)
score, error
linear_regression.coef_
linear_regression.intercept_
pd.DataFrame(linear_regression.coef_, index=df.columns[:-1], columns=['Values'])
# Applying this on test data.

score_test = r2_score(linear_regression.predict(feature_test), label_test)

score_test
score_test*100