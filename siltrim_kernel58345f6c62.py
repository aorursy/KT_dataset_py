import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder





%matplotlib inline

import os

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.describe()
train.info()
dataset = pd.concat((train, test))
dataset = dataset.fillna(np.nan)

dataset.isnull().sum()
g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
g = sns.factorplot(x="SibSp",y='Survived',data=train,kind="bar", size = 6, 

)

g.despine(left=True)

g = g.set_ylabels("survival probability")
g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6,)

g.despine(left=True)

g = g.set_ylabels("survival probability")
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
g = sns.FacetGrid(train, col='Survived')

g = g.map(sns.distplot, "Age")
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].mean())
sns.distplot(dataset['Fare'])

print(dataset["Fare"].skew())
dataset["Fare"] = dataset["Fare"].map(lambda x: np.log(x) if x > 0 else 0)
sns.distplot(dataset['Fare'])

print(dataset["Fare"].skew())
g = sns.barplot(x="Sex",y="Survived",data=train)

g = g.set_ylabel("Survival Probability")
train[["Sex","Survived"]].groupby('Sex').mean()
g = sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar", size = 6)

g.despine(left=True)

g = g.set_ylabels("survival probability")
dataset["Embarked"] = dataset["Embarked"].fillna("S")
g = sns.factorplot(x="Embarked", y="Survived",  data=train,

                   size=6, kind="bar")

g.despine(left=True)

g = g.set_ylabels("survival probability")
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

dataset["Title"] = pd.Series(dataset_title)

dataset["Title"].value_counts()
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

dataset["Title"] = dataset["Title"].astype(int)
g = sns.countplot(dataset["Title"])

g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")

g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])

g = g.set_ylabels("survival probability")
dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1

dataset['Single'] = dataset['FamilySize'].map(lambda s: 1 if s == 1 else 0)

dataset['SmallF'] = dataset['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

dataset['MedF'] = dataset['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

dataset['LargeF'] = dataset['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
dataset['Cabin'].describe()
dataset['Cabin'].isnull().sum()
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
dataset["Ticket"].head()
Ticket = []

for i in list(dataset.Ticket):

    if not i.isdigit() :

        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])

    else:

        Ticket.append("X")

        

dataset["Ticket"] = Ticket

dataset["Ticket"].head()
dataset = pd.get_dummies(dataset, columns = ["Sex", "Title"])

dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")

dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")

dataset["Pclass"] = dataset["Pclass"].astype("category")

dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")

dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")
dataset.drop(labels = ["Name"], axis = 1, inplace = True)

dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)
dataset.head(10)
X_train = dataset[:train.shape[0]]

X_test = dataset[train.shape[0]:]

y = train['Survived']
X_train = X_train.drop(labels='Survived', axis=1)

X_test = X_test.drop(labels='Survived', axis=1)
from sklearn.preprocessing import StandardScaler
headers_train = X_train.columns

headers_test = X_test.columns
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
pd.DataFrame(X_train, columns=headers_train).head()
pd.DataFrame(X_test, columns=headers_test).head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25, random_state = 0 )

accuracies = cross_val_score(LogisticRegression(solver='liblinear'), X_train, y, cv  = cv)

print ("Cross-Validation accuracy scores:{}".format(accuracies))

print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),5)))
from sklearn.model_selection import GridSearchCV
C_vals = [0.2,0.3,0.4,0.5,1,5,10]



penalties = ['l1','l2']



cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25)





param = {'penalty': penalties, 'C': C_vals}



logreg = LogisticRegression(solver='liblinear')

 

grid = GridSearchCV(estimator=LogisticRegression(), 

                           param_grid = param,

                           scoring = 'accuracy',

                            n_jobs =-1,

                           cv = cv

                          )



grid.fit(X_train, y)
print (grid.best_score_)

print (grid.best_params_)

print(grid.best_estimator_)
logreg_grid = grid.best_estimator_

logreg_score = round(logreg_grid.score(X_train,y), 4)

logreg_score
from sklearn.neighbors import KNeighborsClassifier
k_range = range(1,31)

weights_options=['uniform','distance']

param = {'n_neighbors':k_range, 'weights':weights_options}

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

grid = GridSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1)



grid.fit(X_train,y)
print (grid.best_score_)

print (grid.best_params_)

print(grid.best_estimator_)
knn_grid= grid.best_estimator_

knn_score = round(knn_grid.score(X_train,y), 4)

knn_score
from sklearn.naive_bayes import GaussianNB



cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25, random_state = 0 )

accuracies = cross_val_score(GaussianNB(), X_train, y, cv  = cv)

print ("Cross-Validation accuracy scores:{}".format(accuracies))

print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),5)))



bayes_score = round(accuracies.mean(), 4)

bayes_score
from sklearn.svm import SVC



C = [0.1, 1,1.5]

gammas = [0.001, 0.01, 0.1]

kernels = ['rbf', 'poly', 'sigmoid']

param_grid = {'C': C, 'gamma' : gammas, 'kernel' : kernels}



cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=8)



grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=cv)

grid_search.fit(X_train,y)
print(grid_search.best_score_)

print(grid_search.best_params_)

print(grid_search.best_estimator_)
svm_grid = grid_search.best_estimator_

svm_score = round(svm_grid.score(X_train,y), 4)

svm_score
from sklearn.tree import DecisionTreeClassifier

max_depth = range(1,31)

max_feature = [21,22,23,24,25,26,28,29,30,'auto']

criterion=["entropy", "gini"]



param = {'max_depth':max_depth, 

         'max_features':max_feature, 

         'criterion': criterion}



cv=StratifiedShuffleSplit(n_splits=10, test_size =.25, random_state=9)



grid = GridSearchCV(DecisionTreeClassifier(), 

                                param_grid = param, 

                                 verbose=False, 

                                 cv=cv,

                                n_jobs = -1)

grid.fit(X_train, y) 
print( grid.best_params_)

print (grid.best_score_)

print (grid.best_estimator_)
dectree_grid = grid.best_estimator_

dectree_score = round(dectree_grid.score(X_train,y), 4)

dectree_score
import os     

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



from sklearn.externals.six import StringIO

from sklearn.tree import export_graphviz

import pydot

from IPython.display import Image

dot_data = StringIO()  

export_graphviz(dectree_grid, out_file=dot_data,  

                feature_names=headers_train,  class_names = (["Survived" if int(i) is 1 else "Not_survived" for i in y.unique()]),

                filled=True, rounded=True,

                proportion=True,

                special_characters=True)  

(graph,) = pydot.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())
from sklearn.ensemble import RandomForestClassifier

n_estimators = [140,145,150];

max_depth = range(1,10);

criterions = ['gini', 'entropy'];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=10)





parameters = {'n_estimators':n_estimators,

              'max_depth':max_depth,

              'criterion': criterions

              

        }

grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'),

                                 param_grid=parameters,

                                 cv=cv,

                                 n_jobs = -1)

grid.fit(X_train,y)
print (grid.best_score_)

print (grid.best_params_)

print (grid.best_estimator_)
rf_grid = grid.best_estimator_

rf_score = round(rf_grid.score(X_train,y), 4)

rf_score
from xgboost import XGBClassifier
XGBClassifier = XGBClassifier(colsample_bytree = 0.3, subsample = 0.7, reg_lambda = 1)



#colsample_bytree = [0.3, 0.5]

#subsample = [0.7, 1]

n_estimators = [400, 450]

max_depth = [2,3,4]

learning_rate = [0.01, 0.1]

reg_alpha = [0, 0.0001, 0.0005]

#reg_lambda = [0.3, 1, 5]

cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=13)





parameters = {#'colsample_bytree':colsample_bytree,

              #'subsample': subsample,

              'n_estimators':n_estimators,

              'max_depth':max_depth,

              'learning_rate':learning_rate,

              'reg_alpha':reg_alpha,

              #'reg_lambda':reg_lambda

        }

grid = GridSearchCV(estimator=XGBClassifier,

                                 param_grid=parameters,

                                 cv=cv,

                                 n_jobs = -1)

grid.fit(X_train,y)
print (grid.best_score_)

print (grid.best_params_)

print (grid.best_estimator_)
xgb_grid = grid.best_estimator_

xgb_score = round(xgb_grid.score(X_train,y), 4)

xgb_score
from keras.models import Sequential

from keras.layers.core import Dense

import keras

from keras.optimizers import *

from keras.initializers import *
NN_train = dataset[:train.shape[0]]

NN_test = dataset[train.shape[0]:]

NN_y = train['Survived'].values

#NN_y = NN_y.reshape(-1,1)



NN_train = NN_train.drop(labels='Survived', axis=1)

NN_test = NN_test.drop(labels='Survived', axis=1)
sc = StandardScaler()

NN_train = sc.fit_transform(NN_train.values)

NN_test = sc.transform(NN_test.values)
n_cols = NN_train.shape[1]
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(n_cols,)))



model.add(Dense(128, activation="elu"))

model.add(Dense(256, activation="elu"))

model.add(Dense(128, activation="elu"))

model.add(keras.layers.Dropout(0.3))



model.add(Dense(512, activation="elu"))

model.add(Dense(1024, activation="elu"))

model.add(Dense(512, activation="elu"))

model.add(keras.layers.Dropout(0.3))



model.add(Dense(1024, activation="elu"))

model.add(Dense(2048, activation="elu"))

model.add(Dense(1024, activation="elu"))

model.add(keras.layers.Dropout(0.3))



model.add(Dense(512, activation="elu"))

model.add(Dense(1024, activation="elu"))

model.add(Dense(512, activation="elu"))

model.add(keras.layers.Dropout(0.3))



model.add(Dense(256, activation="elu"))

model.add(Dense(128, activation="elu"))

model.add(Dense(64, activation="elu"))

model.add(Dense(32, activation="elu"))

model.add(keras.layers.Dropout(0.3))



model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="Adam", loss='binary_crossentropy', metrics=["binary_accuracy"])
model.summary()
#from keras.callbacks import EarlyStopping

#early_stopping_monitor = EarlyStopping(patience = 10)
model_result = model.fit(NN_train, NN_y, batch_size=100, epochs=200, validation_split = 0.25, shuffle = True)
keras_score = round(max(model_result.history["val_binary_accuracy"]), 4)

keras_score
results = pd.DataFrame({

    'Model': ['Logistic Regression', 'KNN', 'Naive Bayes','Support Vector Machines',   

               'Decision Tree', 'Random Forest', 'XGBoost', 'Keras'],

    'Score': [logreg_score, knn_score, bayes_score, 

              svm_score, dectree_score, rf_score, xgb_score, keras_score]})

results.sort_values(by='Score', ascending=False)

#print df.to_string(index=False)
predict = logreg_grid.predict(X_test)
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predict})

my_submission.to_csv('submission__logreg.csv', index=False)