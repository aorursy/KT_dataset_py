import numpy as np 

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
data = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')
df_x = data.iloc[:,1:]

df_y = data.iloc[:,0]
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
#descision tree

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)
dt.score(x_test,y_test)
dt.score(x_train,y_train)
#Random Forest - Ensemble of Descision Trees



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