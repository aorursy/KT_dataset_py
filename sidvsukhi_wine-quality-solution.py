# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

print(data_df.head())

print(data_df.info())
for column in data_df.columns:

    print('Total number of nan values for column {} are {}'.format(column, data_df[column].isnull().sum()))
print(data_df.describe())

import seaborn as sns



sns.heatmap(data_df.corr())

plt.show()
from sklearn import linear_model

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
X = data_df.loc[:, data_df.columns != 'quality']

X_train, X_test, y_train, y_test = train_test_split(X, data_df['quality'], test_size=0.2, random_state=101)

print("X train \n{}".format(X_train.head()))

print("X test \n{}".format(X_test.head()))

print("y train \n{}".format(y_train.head()))

print("y test \n{}".format(y_test.head()))


from sklearn.metrics import classification_report

#Straight Simple Naive approach

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)

predictions = lm.predict(X_test)

for i in range(len(predictions)):

    predictions[i] = int(round(predictions[i]))

print('Predictions \n{}'.format(predictions))



matches = (predictions == y_test)

print(matches.sum())

print(len(matches))

print('Accuracy {}'.format(matches.sum() / float(len(matches))*100))

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))