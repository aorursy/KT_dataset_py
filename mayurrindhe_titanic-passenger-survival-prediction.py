# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

"","Class","Sex","Age","Survived","Freq"

"1","1st","Male","Child","No",0

"2","2nd","Male","Child","No",0

"3","3rd","Male","Child","No",35

"4","Crew","Male","Child","No",0

"5","1st","Female","Child","No",0

"6","2nd","Female","Child","No",0

"7","3rd","Female","Child","No",17

"8","Crew","Female","Child","No",0

"9","1st","Male","Adult","No",118

"10","2nd","Male","Adult","No",154

"11","3rd","Male","Adult","No",387

"12","Crew","Male","Adult","No",670

"13","1st","Female","Adult","No",4

"14","2nd","Female","Adult","No",13

"15","3rd","Female","Adult","No",89

"16","Crew","Female","Adult","No",3

"17","1st","Male","Child","Yes",5

"18","2nd","Male","Child","Yes",11

"19","3rd","Male","Child","Yes",13

"20","Crew","Male","Child","Yes",0

"21","1st","Female","Child","Yes",1

"22","2nd","Female","Child","Yes",13

"23","3rd","Female","Child","Yes",14

"24","Crew","Female","Child","Yes",0

"25","1st","Male","Adult","Yes",57

"26","2nd","Male","Adult","Yes",14

"27","3rd","Male","Adult","Yes",75

"28","Crew","Male","Adult","Yes",192

"29","1st","Female","Adult","Yes",140

"30","2nd","Female","Adult","Yes",80

"31","3rd","Female","Adult","Yes",76

"32","Crew","Female","Adult","Yes",20



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

%matplotlib inline

tp = pd.read_csv('/kaggle/input/titanic-pass/titanic_pass_sur.csv')

print(tp.columns)
tp.head()
data=tp.dropna()
data.drop(tp.columns[[0]],axis=1,inplace=True)
data.head()
data2=pd.get_dummies(data,columns=['Class','Sex','Age'])

data2.head()
print("dimension of titanic data: {}".format(data2.shape))
X=data2.iloc[:,1:].values

y=data2.iloc[:,0].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data2.loc[:, data2.columns != 'Survived'], data2['Survived'], stratify=data2['Survived'], random_state=66)
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
tree = DecisionTreeClassifier(max_depth=3, random_state=0)

tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
print("Feature importances:\n{}".format(tree.feature_importances_))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=5, random_state=0)

rf.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))