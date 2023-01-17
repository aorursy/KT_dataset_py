# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
cc = pd.read_csv("../input/UCI_Credit_Card.csv")
#PAY_X status : -2 = no usage, -1 = pay duly, 0= revolving credit, 1= pay delay by 1 month ...
#sex (1:male 2:female)
#Education  (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
#Marital status (1=married, 2=single, 3=others)
cc
cc = cc.rename(columns={'default.payment.next.month': 'def_pay', 
                        'PAY_0': 'PAY_1'})
cc.head()
#General probability based on current data
sum(cc['def_pay'])/len(cc['def_pay'])
#22% probability of the default
import seaborn as sns
sns.countplot(x='def_pay', data=cc)
#1 = default
sns.countplot(x='def_pay', hue='SEX', data=cc)
#not much difference
cc['AGE'].plot.hist()
#more young users
cc.groupby(['SEX', 'def_pay']).size()
#more female defaulters
#importing libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split
# split the df into train and test, it is important these two do not communicate during the training
features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
X = cc[features].copy()
X.columns
y = cc['def_pay'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
classifier = DecisionTreeClassifier(max_depth=10, random_state=14) 
# training the classifier
classifier.fit(X_train, y_train)
# do our predictions on the test
predictions = classifier.predict(X_test)
# see how good we did on the test
accuracy_score(y_true = y_test, y_pred = predictions)
