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
df = pd.read_csv('/kaggle/input/diabetes-dataset/diabetes2.csv')
# view first 5 observations

df.head()
# view the shape

df.shape
# select features from the dataset for model building purpose

feature_cols = ['Pregnancies','Insulin','BMI','Age']

# create feature object



x_feature = df[feature_cols]

# create response object

y_target = df['Outcome']
# veiw shape of the feature object

x_feature.shape
# view shape of the target object

y_target.shape
# split test and training data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_feature, y_target, random_state = 1)
# train a logistic regression model on the training set

from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression()

logReg.fit(x_train, y_train)
# make predictions for the testing set

y_pred = logReg.predict(x_test)
# check for accuracy

from sklearn import metrics

print( metrics.accuracy_score(y_test, y_pred))
# print the first 30 true and predicted responses

print( 'actual:', y_test.values[0:30])

print('predicted:', y_pred[0:30])