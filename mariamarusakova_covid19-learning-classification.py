# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
COVID19 = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

COVID19.loc[((COVID19.death != '0') & (COVID19.death != '1')),'death']='1'

COVID19 = COVID19.dropna(subset=['gender', 'age'])

COVID19.drop( COVID19[ COVID19['country'] == 'Sweden' ].index , inplace=True)

train_predictor_columns = ['country','gender', 'age']

X_train, X_test, y_train, y_test = train_test_split(COVID19[['country','gender', 'age']], COVID19['death'], random_state = 0)

print("X_train: {}".format(X_train.shape))

print("X_test: {}".format(X_test.shape))

print("y_train: {}".format(y_train.shape))

print("y_test: {}".format(y_test.shape))

le = LabelEncoder()

le.fit(X_train['gender'].astype(str))



X_train['gender'] = le.transform(X_train['gender'].astype(str))

X_test['gender'] = le.transform(X_test['gender'].astype(str))





le.fit(X_train['country'].astype(str))

X_train['country'] = le.transform(X_train['country'].astype(str))

X_test['country'] = le.transform(X_test['country'].astype(str))
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Predictions for X_test KNeighbors:\n {}".format(y_pred))

print("Reality death cases: {}, reality non-death cases: {}".format(sum(y_test== '1'), sum(y_test== '0')))

print("Predicted death cases: {}, predicted non-death cases: {}".format(sum(y_pred== '1'), sum(y_pred== '0')))

print("Precision when predictions are compared to y_test: {:.2f}".format(np.mean(y_pred == y_test)))