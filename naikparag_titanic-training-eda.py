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
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
train_df

# train_df.info()

# train_df.describe()

# train_df.head(20)

# train_df.tail(10)

# train_df.dtypes
#  what is a dataframe axis ??

cols_to_drop = ['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked']

train_df = train_df.drop(cols_to_drop, axis=1)
train_df
target_df = train_df['Survived']

input_df = train_df.drop(['Survived'], axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(input_df, target_df, test_size=0.33, random_state=42)
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier





def get_model_NB(X, target, alpha):

    model = MultinomialNB(alpha=alpha).fit(X, target)

    return model





def get_model_RF(X, target):

    model = RandomForestClassifier()

    return model.fit(X, target)
model = get_model_RF(X_train, Y_train)
predicted = model.predict(X_test)
# predicted
from sklearn import metrics

print(metrics.classification_report(Y_test, predicted))