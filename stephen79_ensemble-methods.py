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



from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
# load data

train = pd.read_csv('../input/train1.csv')

test1 = pd.read_csv('../input/test.csv')

validation = pd.read_csv('../input/test_to_submit.csv')

train.head()
test1.head()
validation.head()
# shuffle the dataset

train = train.sample(frac=1)

# feature selection

y = train.FraudResult



train = train.drop(['FraudResult'], axis=1)

train.head()
# split dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.1, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



clf = accuracy_score(y_pred,y_test)
# ADABOOST

from sklearn.ensemble import AdaBoostClassifier 

adb = AdaBoostClassifier(n_estimators=100)

adb.fit(X_train, y_train)

y_pred = adb.predict(X_test)



adb = accuracy_score(y_pred,y_test)
# GRADIENT BOOTING

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)

gbc.fit(X_train,y_train)

y_pred = gbc.predict(X_test)



gbc = accuracy_score(y_pred,y_test)
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# comparison of the different models

# using the lasr test fold



plt.figure(figsize=(16,6))

s = sns.barplot(x=['Random Forest','Ada Boost','Gradient Boost'], y=[clf,adb,gbc],color='seagreen')

for p in s.patches:

    s.annotate(format(p.get_height(),".2f"),

               (p.get_x() + p.get_width() /2.,

               

                p.get_height()), ha = "center", va="center",

                xytext = (0,10), textcoords = 'offset points')
# bring real data

# random forest

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



y_to_submit = clf.predict(validation)
submission = pd.DataFrame({'TransactionId':test1['TransactionId'],'FraudResult':y_to_submit})
submission.to_csv('submission3.csv', index=False)