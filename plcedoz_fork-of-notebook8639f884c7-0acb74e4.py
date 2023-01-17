# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train = train.drop(["Ticket", "Cabin", "PassengerId"], axis=1)

test = test.drop(["Ticket", "Cabin"], axis=1)
#Embarked

train['Embarked'] = train['Embarked'].fillna('S')

test['Embarked'] = test['Embarked'].fillna('S')

by_embarked = train.groupby('Embarked')['Survived'].mean()

by_embarked.plot(x=by_embarked.index,y=by_embarked.values, kind='bar', figsize=(5, 3))



#Create Dummies variables

train = train.join(pd.get_dummies(train["Embarked"]))

train = train.drop(['S', 'Embarked'], axis=1)

test = test.join(pd.get_dummies(test["Embarked"]))

test = test.drop(['S', 'Embarked'], axis=1)
#Age for train

old_age = train['Age']

n_null = train['Age'].isnull().sum()

mean_age = train['Age'].mean()

std_age = train['Age'].std()

rand_age = np.random.randint(mean_age-std_age, mean_age+std_age, size = n_null)

train.loc[np.isnan(train['Age']), "Age"] = rand_age



#Age for test

n_null = test['Age'].isnull().sum()

mean_age = test['Age'].mean()

std_age = test['Age'].std()

rand_age = np.random.randint(mean_age-std_age, mean_age+std_age, size = n_null)

test.loc[np.isnan(test['Age']), "Age"] = rand_age



#plot distribution

#plt.figure(1)

#plt.subplot(211)

#plt.hist(old_age, bins = 20)

#plt.subplot(212)

#plt.hist(train['Age'], bins = 20)

#train.info()
#Sex

train.loc[train["Sex"]=="male", "Sex"] = 1

train.loc[train["Sex"]=="female", "Sex"] = 0

test.loc[test["Sex"]=="male", "Sex"] = 1

test.loc[test["Sex"]=="female", "Sex"] = 0
#Family

train["Family"] = (train["SibSp"] + train["Parch"]).copy()

train.loc[train["Family"]==0, "Family"] = 0

train.loc[train["Family"]>0, "Family"] = 1

train = train.drop(["SibSp", "Parch"], axis = 1)





test["Family"] = (test["SibSp"] + test["Parch"]).copy()

test.loc[test["Family"]==0, "Family"] = 0

test.loc[test["Family"]>0, "Family"] = 1

test = test.drop(["SibSp", "Parch"], axis = 1)
#Fare

train["Fare"]=train["Fare"].fillna(train["Fare"].mean())

test["Fare"]=test["Fare"].fillna(test["Fare"].mean())
#Name to title

title = []

for ind in train.index:

    title.append(train["Name"][ind].split(',')[1].split('.')[0])

train["Title"] = pd.Series(title)

by_title = train.groupby("Title")["Age"].count()

for ind in train.index:

    if by_title[train["Title"][ind]] <10:

        train.loc[ind,"Title"] = "rare"

        

avg_title = train.groupby("Title")["Survived"].mean()

avg_title.plot(x=avg_title.index, y=avg_title.values, kind = "bar")        

        

title_2 = []

for ind in test.index:

    title_2.append(test["Name"][ind].split(',')[1].split('.')[0])

test["Title"] = pd.Series(title_2)

by_title = test.groupby("Title")["Age"].count()

for ind in test.index:

    if by_title[test["Title"][ind]] <10:

        test.loc[ind,"Title"] = "rare"

        

#Create dummy variables

title_dummies_train = pd.get_dummies(train["Title"])

title_dummies_train = title_dummies_train.drop([" Mr"], axis = 1)



title_dummies_test = pd.get_dummies(test["Title"])

title_dummies_test = title_dummies_test.drop([" Mr"], axis = 1)



train = train.join(title_dummies_train)

test = test.join(title_dummies_test)



train = train.drop(["Name", "Title"], axis=1)

test = test.drop(["Name", "Title"], axis=1)
#Pclass



by_pclass = train.groupby('Pclass')['Survived'].mean()

by_pclass.plot(x=by_pclass.index, y=by_pclass.values, kind='bar', figsize=(5, 3))



Pclass_dummies = pd.get_dummies(train['Pclass'])

Pclass_dummies.columns = ["Pclass1", "Pclass2", "Pclass3"]

Pclass_dummies = Pclass_dummies.drop(["Pclass3"], axis=1)

train = train.join(Pclass_dummies)

train = train.drop(["Pclass"], axis=1)



Pclass_dummies = pd.get_dummies(test['Pclass'])

Pclass_dummies.columns = ["Pclass1", "Pclass2", "Pclass3"]

Pclass_dummies = Pclass_dummies.drop(["Pclass3"], axis=1)

test = test.join(Pclass_dummies)

test = test.drop(["Pclass"], axis=1)
test.head(10)
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



#Split the dataset

X_train = train.drop(["Survived"], axis=1)

y_train = train["Survived"]

X_test = test.drop ("PassengerId", axis=1).copy()
#Training SVM

from sklearn.svm import SVC

grid = {'kernel' : ['rbf'], 'C' : [1000,10000], 'gamma': [1e-3, 1e-4]}

svc = GridSearchCV(SVC(), grid, cv=5)

svc.fit (X_train, y_train)

params = svc.best_params_

cv_scores = cross_val_score (svc, X_train, y_train, cv = 5)

print(cv_scores, params)

y_svc = svc.predict(X_test)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train, y_train)

y_lr = lr.predict(X_test)

lr.score(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier



grid = {'n_estimators' : [1000], 'max_features' : [2,3]}

rf = GridSearchCV(RandomForestClassifier(), grid, cv=5)

rf.fit (X_train, y_train)

params = rf.best_params_

cv_scores = cross_val_score (rf, X_train, y_train, cv = 5)

print(cv_scores, params)

y_rf = rf.predict(X_test)
from sklearn.neighbors import KNeighborsClassifier



kn = KNeighborsClassifier()

kn.fit(X_train, y_train)

y_kn = kn.predict(X_test)

kn.score(X_train, y_train)
from sklearn.naive_bayes import GaussianNB# Logistic Regression

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_gnb = gnb.predict(X_test)

gnb.score(X_train, y_train)
X_train_nn = X_train.values

y_train_nn = y_train.values

X_test_nn = X_test.values
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier

import numpy





# fix random seed for reproducibility

seed = 7

numpy.random.seed(seed)



# Create Model

def create_model(neurons_layer_1=12, neurons_layer_2=12, dropout=0.2):

    # create model

    model = Sequential()

    model.add(Dense(output_dim = neurons_layer_1, input_dim=12, init='uniform', activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(output_dim = neurons_layer_2, init='uniform', activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # Compile model

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



model = KerasClassifier(build_fn=create_model, nb_epoch=50, batch_size=10, verbose=0)

# define the grid search parameters

neurons_layer_1=[20]

neurons_layer_2=[20]

dropout=[0.25]

grid = dict(neurons_layer_1=neurons_layer_1, neurons_layer_2=neurons_layer_2, dropout=dropout)

nn = GridSearchCV(estimator=model, param_grid=grid)

nn = nn.fit(X_train_nn, y_train_nn)

params = nn.best_params_

cv_scores = cross_val_score (nn, X_train_nn, y_train_nn, cv = 5)

print(params, cv_scores)

#Make predictions

y_nn = nn.predict(X_test_nn)

y_nn[y_nn>0.5]=1

y_nn[y_nn<0.5]=0

y_nn = y_nn[:,0]
cv_scores
y_test = test["PassengerId"]

solution_nn = pd.DataFrame({

    "PassengerId" : y_test,

    "Survived" : y_nn[:,0]

})

solution_nn.to_csv("solution_nn.csv", index=False)