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
import numpy as np 

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data.head()
df_x = data.iloc[:,1:]

df_y = data.iloc[:,0]
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)


#descision tree

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)
dt.score(x_test,y_test)
rf = RandomForestClassifier(n_estimators=20)

rf.fit(x_train,y_train)
rf.score(x_test,y_test)
#Bagging 



bg = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)

bg.fit(x_train,y_train)
bg.score(x_test,y_test)
bg.score(x_train,y_train)
#Boosting - Ada Boost



adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 5, learning_rate = 1)

adb.fit(x_train,y_train)
adb.score(x_test,y_test)
adb.score(x_train,y_train)
# Voting Classifier - Multiple Model Ensemble 



lr = LogisticRegression()

dt = DecisionTreeClassifier()

svm = SVC(kernel = 'poly', degree = 2 )
evc = VotingClassifier( estimators= [('lr',lr),('dt',dt),('svm',svm)], voting = 'hard')
evc.fit(x_train.iloc[1:4000],y_train.iloc[1:4000])
evc.score(x_test, y_test)