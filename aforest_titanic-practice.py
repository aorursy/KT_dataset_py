# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

from pandas import Series, DataFrame



from PIL import Image

#from sklearn.cross_validation import train_test_split, cross_val_score, KFold

#from sklearn.cross_validation import *

from sklearn.linear_model import LogisticRegression as Classifier1

from sklearn.metrics import accuracy_score

#from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.ensemble import RandomForestClassifier as Classifier2

from sklearn.ensemble import AdaBoostClassifier as Classifier3

from sklearn.neighbors import KNeighborsClassifier as Classifier4

from sklearn.ensemble import ExtraTreesClassifier as Classifier5

from sklearn.svm import SVC as Classifier6

from sklearn.svm import LinearSVC as Classifier7

from sklearn.ensemble import GradientBoostingClassifier as Classifier8

from sklearn.ensemble import BaggingClassifier as Classifier9
data = pd.read_csv("../input/train.csv")

data.head()
data2 = pd.read_csv("../input/test.csv")
tmp = data[['Age','Sex','Pclass','Fare','Survived','SibSp','Parch','Ticket','Embarked','Cabin','Name']]

X = tmp[['Age','Sex','Pclass','Fare','SibSp','Parch','Embarked','Cabin']]

X['gender'] = X['Sex'].map({'female':0,'male':1}).astype(int)

X = X.drop(['Sex'], axis=1)



X['emS'] = X['Embarked'].map({'S':1,'Q':0,'C':0,np.nan:0}).astype(int)

X['emC'] = X['Embarked'].map({'S':0,'Q':0,'C':1,np.nan:0}).astype(int)

X['emQ'] = X['Embarked'].map({'S':0,'Q':1,'C':0,np.nan:0}).astype(int)

X = X.drop(['Embarked'], axis=1)



# replacing missing cabins with U (for Uknown)

X[ 'Cabin' ] = X.Cabin.fillna('U')

X[ 'Cabin' ] = X[ 'Cabin' ].map( lambda c : c[0] )

X['caA'] = X['Cabin'].map({'A':1,'B':0,'C':0,'D':0,'E':0,'F':0,'G':0,'T':0,'U':0}).astype(int)

X['caB'] = X['Cabin'].map({'A':0,'B':1,'C':0,'D':0,'E':0,'F':0,'G':0,'T':0,'U':0}).astype(int)

X['caC'] = X['Cabin'].map({'A':0,'B':0,'C':1,'D':0,'E':0,'F':0,'G':0,'T':0,'U':0}).astype(int)

X['caD'] = X['Cabin'].map({'A':0,'B':0,'C':0,'D':1,'E':0,'F':0,'G':0,'T':0,'U':0}).astype(int)

X['caE'] = X['Cabin'].map({'A':0,'B':0,'C':0,'D':0,'E':1,'F':0,'G':0,'T':0,'U':0}).astype(int)

X['caF'] = X['Cabin'].map({'A':0,'B':0,'C':0,'D':0,'E':0,'F':1,'G':0,'T':0,'U':0}).astype(int)

X['caG'] = X['Cabin'].map({'A':0,'B':0,'C':0,'D':0,'E':0,'F':0,'G':1,'T':0,'U':0}).astype(int)

X['caT'] = X['Cabin'].map({'A':0,'B':0,'C':0,'D':0,'E':0,'F':0,'G':0,'T':1,'U':0}).astype(int)

X = X.drop(['Cabin'], axis=1)



X = X.fillna(X.mean())

X = X.as_matrix()



y = tmp['Survived']

X
tmp = data2[['Age','Sex','Pclass','Fare','SibSp','Parch','Ticket','Embarked','Cabin','Name']]

Test = tmp[['Age','Sex','Pclass','Fare','SibSp','Parch','Embarked','Cabin']]

Test['gender'] = Test['Sex'].map({'female':0,'male':1}).astype(int)

Test = Test.drop(['Sex'], axis=1)

Test['emS'] = Test['Embarked'].map({'S':1,'Q':0,'C':0,np.nan:0}).astype(int)

Test['emC'] = Test['Embarked'].map({'S':0,'Q':0,'C':1,np.nan:0}).astype(int)

Test['emQ'] = Test['Embarked'].map({'S':0,'Q':1,'C':0,np.nan:0}).astype(int)

Test = Test.drop(['Embarked'], axis=1)



# replacing missing cabins with U (for Uknown)

Test[ 'Cabin' ] = Test.Cabin.fillna('U')

Test[ 'Cabin' ] = Test[ 'Cabin' ].map( lambda c : c[0] )

Test['caA'] = Test['Cabin'].map({'A':1,'B':0,'C':0,'D':0,'E':0,'F':0,'G':0,'T':0,'U':0}).astype(int)

Test['caB'] = Test['Cabin'].map({'A':0,'B':1,'C':0,'D':0,'E':0,'F':0,'G':0,'T':0,'U':0}).astype(int)

Test['caC'] = Test['Cabin'].map({'A':0,'B':0,'C':1,'D':0,'E':0,'F':0,'G':0,'T':0,'U':0}).astype(int)

Test['caD'] = Test['Cabin'].map({'A':0,'B':0,'C':0,'D':1,'E':0,'F':0,'G':0,'T':0,'U':0}).astype(int)

Test['caE'] = Test['Cabin'].map({'A':0,'B':0,'C':0,'D':0,'E':1,'F':0,'G':0,'T':0,'U':0}).astype(int)

Test['caF'] = Test['Cabin'].map({'A':0,'B':0,'C':0,'D':0,'E':0,'F':1,'G':0,'T':0,'U':0}).astype(int)

Test['caG'] = Test['Cabin'].map({'A':0,'B':0,'C':0,'D':0,'E':0,'F':0,'G':1,'T':0,'U':0}).astype(int)

Test['caT'] = Test['Cabin'].map({'A':0,'B':0,'C':0,'D':0,'E':0,'F':0,'G':0,'T':1,'U':0}).astype(int)

Test = Test.drop(['Cabin'], axis=1)







#Test = Test.fillna({'Age':20,'Pclass':3,'Fare':7,'genr':1})

Test = Test.fillna(Test.mean())

Test = Test.as_matrix()
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Activation, Dropout

from keras.utils.np_utils import to_categorical
model = Sequential()



model.add(Dense(200, input_dim= 17))

model.add(Activation("relu"))

model.add(Dropout(0.5))



model.add(Dense(200))

model.add(Activation("relu"))

model.add(Dropout(0.5))



model.add(Dense(200))

model.add(Activation("relu"))

model.add(Dropout(0.5))



model.add(Dense(2))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



label_train_category = to_categorical(y)

#label_test_category = to_categorical(Test)





model.fit(X, label_train_category, nb_epoch=4000, batch_size=100, verbose=0)



results = model.predict_classes(Test, verbose=1)
submit_csv1 = pd.DataFrame(data2['PassengerId'])

submit_csv2 = pd.DataFrame()



submit_csv2['Survived'] = results





submit_csv = pd.concat([submit_csv1, submit_csv2], axis=1)



submit_csv
submit_csv.to_csv('output.csv',header=True, index=False)