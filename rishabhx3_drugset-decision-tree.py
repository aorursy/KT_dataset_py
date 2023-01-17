import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn import metrics

import matplotlib.image as mpimg



%matplotlib inline 
data = pd.read_csv('../input/drugsets/drug200.csv')

data.head()
X = data[['Age','Sex','BP','Cholesterol','Na_to_K']].values

y = data[['Drug']].values #Dependent variable
le_sex = preprocessing.LabelEncoder()

le_sex.fit(['F','M'])

X[:,1] = le_sex.transform(X[:,1])



le_BP = preprocessing.LabelEncoder()

le_BP.fit(['LOW','NORMAL','HIGH'])

X[:,2] = le_BP.transform(X[:,2])



le_Chol = preprocessing.LabelEncoder()

le_Chol.fit([ 'NORMAL', 'HIGH'])

X[:,3] = le_Chol.transform(X[:,3]) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
model.fit(X_train,y_train)
predTree = model.predict(X_test)
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_test, predTree))