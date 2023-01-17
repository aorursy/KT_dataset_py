#Modules neeeded
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fancyimpute import KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#Import data files

train = pd.read_csv("../input/train.csv") # pd at the begining refers to pandas module
test = pd.read_csv("../input/test.csv")
#Data description

print ("Total rows in train dataset = ", len(train))
print ("Data type in train =", train.dtypes)
print("********************************************")
print ("Total rows in test dataset = ", len(test))
print ("Data type in test =", test.dtypes)
#Data quickview

train.head()
#Data summary

train.describe(include='all')
test.describe(include='all')
x = ["Age","Cabin"]
complete = [train["Age"].count(), train["Cabin"].count() ] 
missings = [sum(train["Age"].isnull()), sum(train["Cabin"].isnull())] 

p1 = plt.bar(x , complete)
p2 = plt.bar(x , missings)


plt.show()
#Dropping cabin

train.drop(['Cabin'],axis=1, inplace=True)
test.drop(['Cabin'],axis=1, inplace=True)
## create dummy variables for Column sex and embarked since they are categorical value
## Renaming the Sex_male column to Gender, since this will be quite important part of this notebook later on. 
train = pd.get_dummies(train, columns=['Embarked'])
train = pd.get_dummies(train, columns=['Sex'], drop_first=True)
train = train.rename(columns={"Sex_male": "Gender"})


test = pd.get_dummies(test, columns=['Embarked'])
test = pd.get_dummies(test, columns=['Sex'], drop_first=True)
test = test.rename(columns={"Sex_male": "Gender"})

train.drop(['Name', 'Ticket'],axis=1, inplace=True)
test.drop(['Name', 'Ticket'],axis=1, inplace=True)
#Replace missings according to similar characteristics

age_train = KNN(k=80).complete(train)

train = pd.DataFrame(age_train, columns = train.columns)
#Lets take a look at the result

train.head()
test_index = test.index.values
# importing missing values using KNN for age column. 

#why k = 10? 
age_test = KNN(k=80).complete(test)

test = pd.DataFrame(age_test, columns = test.columns)
survived_summary = train.groupby("Survived")
survived_summary.describe(include='all')
#identifying target variable

X = train.drop(['Survived'], axis=1)
y = train["Survived"]

#Split training data

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = .33, random_state = 1)
knn = KNeighborsClassifier(weights="uniform", )
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
knn_accy = round(accuracy_score(y_test, y_pred), 3)
print (knn_accy)
#KNN grid search
knn = KNeighborsClassifier()
n_neighbors=range(1,50)
weights=['uniform','distance']
param = {'n_neighbors':n_neighbors, 'weights':weights}
grid2 = GridSearchCV(knn, param,verbose=False, cv=StratifiedKFold(n_splits=5, random_state=15, shuffle=True))
grid2.fit(x_train, y_train)
print (grid2.best_params_)
print (grid2.best_score_)
knn_grid = KNeighborsClassifier(
    n_neighbors = grid2.best_params_['n_neighbors'], 
    weights = grid2.best_params_['weights'],
    n_jobs = 1, 
)
knn_grid.fit(x_train,y_train)
y_pred = knn_grid.predict(x_test)
knn_accy = round(accuracy_score(y_test, y_pred), 3)
print (knn_accy)
model = XGBClassifier()
model.fit(x_train, y_train)
# make predictions for test data

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
# accurracy

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
