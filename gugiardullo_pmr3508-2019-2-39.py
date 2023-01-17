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
train_data = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv', na_values = "?")
from sklearn import preprocessing

nadult = train_data.dropna()
nadult.shape
attributes = ["age", "workclass", "fnlwgt", "education.num","relationship", "race", "sex", "capital.gain", "capital.loss", "hours.per.week"]
train_data.info()
train_adult_x = nadult[attributes].apply(preprocessing.LabelEncoder().fit_transform)
train_adult_x.head()
train_adult_y = nadult.income
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
n = train_data['Id'].nunique()

p = np.sqrt(n)

p
rfc = RandomForestClassifier(n_estimators=180)
scores = cross_val_score(rfc, train_adult_x, train_adult_y, cv = 10)

scores
test_data =  pd.read_csv('/kaggle/input/adult-pmr3508/test_data.csv')
test_adult_x = test_data[attributes].apply(preprocessing.LabelEncoder().fit_transform)
rfc.fit(train_adult_x,train_adult_y)

test_pred_y = rfc.predict(test_adult_x)

test_pred_y
id_index = pd.DataFrame({'Id' : list(range(len(test_pred_y)))})

income = pd.DataFrame({'income' : test_pred_y})

result = id_index.join(income)

result
result.to_csv("submission_random_forest.csv", index = False)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
scores = cross_val_score(logmodel, train_adult_x, train_adult_y, cv = 10)

scores
from sklearn.svm import SVC

svc_model = SVC()
scores = cross_val_score(svc_model, train_adult_x, train_adult_y, cv = 10)

scores