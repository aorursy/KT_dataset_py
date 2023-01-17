import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

train = pd.read_csv("../input/random-linear-regression/train.csv") 

test = pd.read_csv("../input/random-linear-regression/test.csv") 

train = train.dropna()

test = test.dropna()

train.head()



# Model PLot and Accuracy

X_train = np.array(train.iloc[:, :-1].values)

y_train = np.array(train.iloc[:, 1].values)

X_test = np.array(test.iloc[:, :-1].values)

y_test = np.array(test.iloc[:, 1].values)

model = LinearRegression()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)



plt.plot(X_train, model.predict(X_train), color='green')

plt.show()

print(accuracy)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

#Data is used the same as LGB

X = train.drop(columns=['item_price', 'item_id']) 

y = train['item_price']

X.head()



# Model & Accuracy

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

lasso.fit(X, y)

r2_score(lasso.predict(X), y)
import sklearn

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import r2_score

from statistics import mode





train = pd.read_csv("../input/titanic/train.csv")

test  = pd.read_csv('../input/titanic/test.csv')

train.head()





ports = pd.get_dummies(train.Embarked , prefix='Embarked')

train = train.join(ports)

train.drop(['Embarked'], axis=1, inplace=True)

train.Sex = train.Sex.map({'male':0, 'female':1})

y = train.Survived.copy()

X = train.drop(['Survived'], axis=1) 

X.drop(['Cabin'], axis=1, inplace=True) 

X.drop(['Ticket'], axis=1, inplace=True) 

X.drop(['Name'], axis=1, inplace=True) 

X.drop(['PassengerId'], axis=1, inplace=True)

X.Age.fillna(X.Age.median(), inplace=True)



#Model and Accuracy



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter = 500000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)

print(accuracy)
# Support Vector Machine
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

data_svm = pd.read_csv("../input/svm-classification/UniversalBank.csv")

data_svm.head()







#model & accuuracy



X = data_svm.iloc[:,1:13].values

y = data_svm.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

accuracies.mean()