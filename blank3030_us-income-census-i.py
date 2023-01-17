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

import numpy as np

import matplotlib.pyplot as plt
training_data = '../input/adult-training.csv'
test_data = '../input/adult-test.csv'
columns = ['Age', 'Workclass', 'fnlgwt', 'Education', 'Education num', 'Martial status', 'occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hour/week', 'Country', 'Above/below 50K']
train = pd.read_csv(training_data, names = columns)

test = pd.read_csv(test_data, names = columns)
train.head()
test.head()
test = test.drop(test.index[0]).head()
def missing_value(df):

    miss = []

    col_list = df.columns

    for i in col_list:

        missing = df[i].isnull().sum()

        miss.append(missing)

        list_of_missing = pd.DataFrame(list(zip(col_list, miss)))

    return list_of_missing

missing_value(train)
missing_value(test)
train.apply(lambda x: len(x.unique()))
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder
test['Workclass'].value_counts()
train['Workclass'] =train['Workclass'].str.replace('?', 'Private')

test['Workclass'] =test['Workclass'].str.replace('?', 'Private')
train['occupation'].value_counts()
train['occupation'] = train['occupation'].str.replace('?', 'Prof-specialty')
test['occupation'] = test['occupation'].str.replace('?', 'Prof-specialty')
test['Above/below 50K'] = test['Above/below 50K'].str.replace('K.', 'K')
train['Sex'].value_counts()
train_uni = train.apply(lambda x: len(x.unique()))
train_uni
train_uni['Race']
X_train = train.iloc[:, :14]

y_train = train.iloc[:, 14]

X_test = test.iloc[:, :-1]

y_test = test.iloc[:, -1]
X_train.head()
col = ['Workclass', 'Education', 'Education num', 'Martial status', 'occupation', 'Relationship', 'Race', 'Sex', 'Country']
col_mask = X_train.dtypes==object 
col_cat = [x for x in train.dtypes.index if train.dtypes[x] == 'object']
col_mask
for col in train.columns:

    if train.dtypes[col] == 'object':

        if col != 'Above/below 50K':

            le = LabelEncoder()

            X_train[col] = le.fit_transform(X_train[col])

            X_test[col] = le.transform(X_test[col])

le1 = LabelEncoder()

y_train = le1.fit_transform(y_train)

y_test = le1.transform(y_test)
ohe = OneHotEncoder(categorical_features = col_mask, sparse = False)

train_ohe = ohe.fit_transform(X_train)

test_ohe = ohe.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C = 1e7, random_state = 0)
classifier.fit(train_ohe, y_train)
pred1 = classifier.predict(test_ohe)
pred1
classifier.coef_
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred1)

print(cm)
plt.scatter(X_test['Age'],pred1)
from sklearn.neighbors import KNeighborsClassifier

clf2 = KNeighborsClassifier(n_neighbors = 2)

clf2.fit(X_train, y_train)

pred2 = clf2.predict(X_test)

cm2 = confusion_matrix(y_test, pred2)
pred2
from sklearn.ensemble import RandomForestClassifier

clf3 = RandomForestClassifier()

clf3.fit(train_ohe, y_train)

pred3 = clf3.predict(test_ohe)

cm3 = confusion_matrix(y_test, pred3)
pred3
from sklearn.feature_selection import SelectFromModel

sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))

sel.fit(X_train, y_train)

sel.get_support()

selected_feat = X_train.columns[(sel.get_support())]

selected_feat
new_X_train = pd.DataFrame()

new_X_test = pd.DataFrame()
new_X_train['Age'] = X_train['Age']

new_X_train['fnlgwt'] = X_train['fnlgwt']

new_X_train['Education num'] = X_train['Education num']

new_X_train['Relationship'] = X_train['Relationship']

new_X_train['Capital Gain'] = X_train['Capital Gain']

new_X_train['Hour/week'] = X_train['Hour/week']
new_X_test['Age'] = X_test['Age']

new_X_test['fnlgwt'] = X_test['fnlgwt']

new_X_test['Education num'] = X_test['Education num']

new_X_test['Relationship'] = X_test['Relationship']

new_X_test['Capital Gain'] = X_test['Capital Gain']

new_X_test['Hour/week'] = X_test['Hour/week']
new_X_train.head()
clf4 = RandomForestClassifier()

clf4.fit(new_X_train, y_train)

pred4 = clf4.predict(new_X_test)

cm4 = confusion_matrix(y_test, pred4)
pred4