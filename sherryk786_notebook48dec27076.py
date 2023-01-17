# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import sklearn

from sklearn import metrics, model_selection

import scipy,re,os,sklearn

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn import model_selection

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/HR_comma_sep.csv")

left = data['left']

rest = data.drop(['left', 'sales'], 1)

rest['salary'] = rest['salary'].map({'low':0,'medium':1,'high':2})

promotion = data['promotion_last_5years']

print("---------")

print("Sample of the input data, columns")

print("---------")

print(data[:5])

print("---------")

print("List of all types of positions:- ", data['sales'].unique())

print("---------")

print("List of all types of salary:- ", data['salary'].unique())

print("---------")

print (sum(promotion.values) , " personnel promoted in the last 5 years.")

print("---------")

left_train, left_test, rest_train, rest_test = model_selection.train_test_split(left, rest, test_size = 0.3)
model = DecisionTreeClassifier(random_state=0)

model.fit(rest_train, left_train)

print (model.score(rest_test, left_test))