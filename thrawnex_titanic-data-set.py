# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Importing of Train Data
data=pd.read_csv('../input/train.csv')
data.head()

data.isnull().sum()
len(data['Cabin'])

#Feature Selection
# Unnötige Features: Namen auf jeden Fall
data.drop('Name', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)
data.drop('Cabin', axis=1, inplace=True)
data.drop('PassengerId', axis=1, inplace=True)

data.head()
#Sex encoden
data['Sex'].replace(['male','female'],[0,1],inplace=True)

# Fehlende Ages durch den Mittelwert ersetzen.
data['Age'].fillna(data['Age'].mean(),inplace=True)



data['Embarked'].fillna('S',inplace=True)

data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

data.isnull().sum()


train, test = train_test_split(data, test_size=0.2, random_state=0, stratify=data['Survived'])
train_y = train['Survived']
train_X = train.drop('Survived', axis=1)
test_y = train['Survived']
test_X = train.drop('Survived', axis=1)

clf = LogisticRegression(C=30000, solver='lbfgs', tol=1e-6, max_iter=500, multi_class='multinomial')


clf.fit(train_X, train_y)
prediction=clf.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_y))
from sklearn.neural_network import MLPClassifier
clf2 = MLPClassifier()

clf2.fit(train_X, train_y)
prediction2=clf2.predict(test_X)
print('The accuracy of the MLP is',metrics.accuracy_score(prediction2,test_y))
test=pd.read_csv('../input/test.csv')
#Drop Name, Ticket, Cabin, PassengerID

test.drop('Name', axis=1, inplace=True)
test.drop('Ticket', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)
passenger = test['PassengerId']
test.drop('PassengerId', axis=1, inplace=True)


# Change Categorial Features to Numbers 
# Fill NaNs 
test['Sex'].replace(['male','female'],[0,1],inplace=True)
test['Age'].fillna(data['Age'].mean(),inplace=True)
test['Embarked'].fillna('S',inplace=True)
test['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
test.isnull().sum()

passenger.head()
y = test['Survived']
X = test.drop('Survived', axis=1)
prediction=clf.predict(X)
prediction2=clf2.predict(X)
print('The accuracy of the Logistic Regression on Test Set is',metrics.accuracy_score(prediction,y))
print('The accuracy of the MLP on Test Set is',metrics.accuracy_score(prediction2,y))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, prediction2)
print(cm)
#from sklearn.model_selection import GridSearchCV



#learning_rate = ['constant', 'invscaling', 'adaptive']
#solver=['adam','lbfgs', 'sgd']
#max_iter=[200, 250, 300, 400, 500]
#hidden_layer_sizes = [(100,1), (100,2), (100,3)]
#print("Starting Optimisation")
#hyper={'solver':solver, 'learning_rate':learning_rate, 'max_iter': max_iter, 'hidden_layer_sizes': hidden_layer_sizes}
#gd=GridSearchCV(estimator=MLPClassifier(),param_grid=hyper,verbose=True, n_jobs=8)
#gd.fit(train_X, train_y)
#print(gd.best_score_)
#print(gd.best_estimator_)

prediction2.reshape(891)
df = pd.DataFrame({'passengerid': passenger, 'prediction':prediction2})
string = "Gruppe.csv"
df.shape
df.to_csv(string, index=False)



