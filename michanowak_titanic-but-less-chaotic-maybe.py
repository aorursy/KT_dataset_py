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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from warnings import filterwarnings

filterwarnings("ignore")

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.compose import ColumnTransformer
train.head()
train.isnull().sum()
train = train.drop(["PassengerId"], axis=1)

train = train.drop(["Survived"], axis=1)

train = train.drop(["Ticket"], axis=1)
train.Embarked = train.Embarked.fillna('S')
train['Title'] = train.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
normalized_titles = {

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"

}

train.Title = train.Title.map(normalized_titles)

train = train.drop(["Name"], axis=1)



Title_Medians = train.groupby('Title').median()

Title_Medians.Age
train.Title.value_counts()
#train.loc[train['Title'] == "Officer"].Age = train.loc[train['Title'] == "Officer"].Age.fillna(train.loc[train['Title'] == 'Officer']["Age"].mean(skipna = True))
#MrT = train.loc[train['Title'] == "Mr"]

#MissT = train.loc[train['Title'] == "Miss"]

#MrsT = train.loc[train['Title'] == "Mrs"]

#MasterT = train.loc[train['Title'] == "Master"]

#OfficerT = train.loc[train['Title'] == "Officer"]

#RoyaltyT = train.loc[train['Title'] == "Royalty"]
#MrT.Age = MrT.Age.fillna(Title_Medians["Age"]["Mr"])

#MissT.Age = MissT.Age.fillna(Title_Medians["Age"]["Miss"])

#MrsT.Age = MrsT.Age.fillna(Title_Medians["Age"]["Mrs"])

#MasterT.Age = MasterT.Age.fillna(Title_Medians["Age"]["Master"])

#OfficerT.Age = OfficerT.Age.fillna(Title_Medians["Age"]["Officer"])

#RoyaltyT.Age = RoyaltyT.Age.fillna(Title_Medians["Age"]["Royalty"])
#X = pd.concat([MrT,MissT,MrsT,MasterT,OfficerT,RoyaltyT])

#train = X.sort_index()

#train
grouped = train.groupby(['Title'])  

grouped
train.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
train.Cabin = train.Cabin.fillna('U')

train.Cabin = train.Cabin.str.slice(0,1)

train["Cabin"].value_counts()
train.head()
corr = train.corr()

corr["Age"].sort_values(ascending=False)
def data_preparation(X):

    

    X = X.drop(["Ticket","PassengerId"], axis=1)

    

    X.Embarked = X.Embarked.fillna('S')

    X.Fare = X.Fare.fillna(X.Fare.median())

    

    X['Title'] = X.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

    normalized_titles = {

        "Capt":       "Officer",

        "Col":        "Officer",

        "Major":      "Officer",

        "Jonkheer":   "Royalty",

        "Don":        "Royalty",

        "Sir" :       "Royalty",

        "Dr":         "Officer",

        "Rev":        "Officer",

        "the Countess":"Royalty",

        "Dona":       "Royalty",

        "Mme":        "Mrs",

        "Mlle":       "Miss",

        "Ms":         "Mrs",

        "Mr" :        "Mr",

        "Mrs" :       "Mrs",

        "Miss" :      "Miss",

        "Master" :    "Master",

        "Lady" :      "Royalty"

    }

    X.Title = X.Title.map(normalized_titles)

    X = X.drop(["Name"], axis=1)

    

    grouped = X.groupby(['Title'])  

    X.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))

    

    X.Cabin = X.Cabin.fillna('U')

    X.Cabin = X.Cabin.str.slice(0,1)

    

    return X
from sklearn.preprocessing import OneHotEncoder



num_features = ["Age", "Fare", "SibSp", "Parch"]

num_pipeline = make_pipeline(StandardScaler())



cat_features = ["Pclass", "Sex", "Cabin", "Embarked", "Title"]

cat_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))



preprocessor = ColumnTransformer(transformers=[

    ("num", num_pipeline, num_features),

    ("cat", cat_pipeline, cat_features)

])
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
Xpreprocessed = data_preparation(train)



X_train, X_test, y_train, y_test = train_test_split(Xpreprocessed,Xpreprocessed.Survived,test_size=100, random_state = 50)

X_test = X_test.drop(["Survived"], axis = 1)

X_train = X_train.drop(["Survived"], axis = 1)



model = RandomForestClassifier()

classification_process = make_pipeline(preprocessor, model)

classification_process.fit(X_train, y_train)



accuracy_score(y_test, classification_process.predict(X_test))
X_test = data_preparation(test)

X_train = data_preparation(train)

y_train = X_train.Survived

X_train = X_train.drop(["Survived"], axis = 1)
classification_process.fit(X_train, y_train)

predictions = classification_process.predict(X_test)

result = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

#result.to_csv('first_result.csv', index=False)

print(result)
train["Parch"].value_counts()
train["SibSp"].value_counts()
X_train = data_preparation(train)
features = ["Pclass", "Sex", "SibSp", "Parch", "Cabin", "Embarked", "Title", "Survived", "Fare", "Age"]

corrcheck = pd.get_dummies(X_train[features])

corrcheck.head()
corr = corrcheck.corr()

corr["Survived"].sort_values(ascending=False)
X_train["Title"].value_counts()
plt.scatter(X_train["Age"], X_train["Survived"])

plt.show()
plt.scatter(X_train["Fare"], X_train["Survived"])

plt.show()
plt.scatter(X_train["SibSp"], X_train["Survived"])

plt.show()
plt.scatter(X_train["Parch"], X_train["Survived"])

plt.show()
plt.scatter(X_train["Age"], X_train["Fare"], c = X_train["Survived"])

plt.show()
plt.scatter(X_train["Title"], X_train["Age"], c = X_train["Survived"])

plt.show()
plt.scatter(X_train["Cabin"], X_train["Age"], c = X_train["Survived"])

plt.show()
X_train.Cabin = X_train.Cabin.replace("T","U")

X_train.Cabin = X_train.Cabin.replace("U","unknown")

X_train.Cabin = X_train.Cabin.replace("C","known")

X_train.Cabin = X_train.Cabin.replace("E","known")

X_train.Cabin = X_train.Cabin.replace("G","known")

X_train.Cabin = X_train.Cabin.replace("D","known")

X_train.Cabin = X_train.Cabin.replace("A","known")

X_train.Cabin = X_train.Cabin.replace("B","known")

X_train.Cabin = X_train.Cabin.replace("F","known")
bins = [-1, 0.5, 1.5, np.inf]

labels = ['0', '1', '2+']

X_train['SibSp'] = pd.cut(X_train["SibSp"], bins, labels = labels)
bins = [-1, 0.5, 1.5, np.inf]

labels = ['0', '1', '2+']

X_train['Parch'] = pd.cut(X_train["Parch"], bins, labels = labels)
features = ["Pclass", "Sex", "SibSp", "Parch", "Cabin", "Embarked", "Title", "Survived", "Fare", "Age"]

corrcheck = pd.get_dummies(X_train[features])

corr = corrcheck.corr()

corr["Survived"].sort_values(ascending=False)
num_features = ["Fare"]

num_pipeline = make_pipeline(StandardScaler())



cat_features = ["Pclass", "Cabin", "Title", "SibSp", "Parch", "Embarked", "Sex"]

cat_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))



preprocessor = ColumnTransformer(transformers=[

    ("num", num_pipeline, num_features),

    ("cat", cat_pipeline, cat_features)

])
Xpreprocessed = X_train



X_train, X_test, y_train, y_test = train_test_split(Xpreprocessed,Xpreprocessed.Survived,test_size=100, random_state = 50)

X_test = X_test.drop(["Survived"], axis = 1)

X_train = X_train.drop(["Survived"], axis = 1)



model = RandomForestClassifier()

classification_process = make_pipeline(preprocessor, model)

classification_process.fit(X_train, y_train)



accuracy_score(y_test, classification_process.predict(X_test))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
num_features = ["Age", "Fare"]

num_pipeline = make_pipeline(StandardScaler())



cat_features = ["Pclass", "Cabin", "Title", "SibSp", "Parch", "Embarked"]

cat_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))



preprocessor = ColumnTransformer(transformers=[

    ("num", num_pipeline, num_features),

    ("cat", cat_pipeline, cat_features)

])
def data_preparation(X):

    

    X = X.drop(["Ticket","PassengerId"], axis=1)

    

    X.Embarked = X.Embarked.fillna('S')

    X.Fare = X.Fare.fillna(X.Fare.median())

    

    X['Title'] = X.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

    normalized_titles = {

        "Capt":       "Officer",

        "Col":        "Officer",

        "Major":      "Officer",

        "Jonkheer":   "Royalty",

        "Don":        "Royalty",

        "Sir" :       "Royalty",

        "Dr":         "Officer",

        "Rev":        "Officer",

        "the Countess":"Royalty",

        "Dona":       "Royalty",

        "Mme":        "Mrs",

        "Mlle":       "Miss",

        "Ms":         "Mrs",

        "Mr" :        "Mr",

        "Mrs" :       "Mrs",

        "Miss" :      "Miss",

        "Master" :    "Master",

        "Lady" :      "Royalty"

    }

    X.Title = X.Title.map(normalized_titles)

    X = X.drop(["Name"], axis=1)

    

    grouped = X.groupby(['Title'])  

    X.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))

      

    X.Cabin = X.Cabin.str.slice(0,1)

    X.Cabin = X.Cabin.fillna("unknown")

    X.Cabin = X.Cabin.replace("T","unknown")

    X.Cabin = X.Cabin.replace("C","known")

    X.Cabin = X.Cabin.replace("E","known")

    X.Cabin = X.Cabin.replace("G","known")

    X.Cabin = X.Cabin.replace("D","known")

    X.Cabin = X.Cabin.replace("A","known")

    X.Cabin = X.Cabin.replace("B","known")

    X.Cabin = X.Cabin.replace("F","known")

    

    bins = [-1, 0.5, 1.5, np.inf]

    labels = ['0', '1', '2+']

    X['SibSp'] = pd.cut(X["SibSp"], bins, labels = labels)

    

    bins = [-1, 0.5, 1.5, np.inf]

    labels = ['0', '1', '2+']

    X['Parch'] = pd.cut(X["Parch"], bins, labels = labels)

    

    

    return X
X_test = data_preparation(test)

X_train = data_preparation(train)

y_train = X_train.Survived

X_train = X_train.drop(["Survived"], axis = 1)



model = RandomForestClassifier()

classification_process = make_pipeline(preprocessor, model)

classification_process.fit(X_train, y_train)
classification_process.fit(X_train, y_train)

predictions = classification_process.predict(X_test)

result = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

#result.to_csv('second_result.csv', index=False)

print(result)
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
num_features = ["Fare"]

num_pipeline = make_pipeline(StandardScaler())



cat_features = ["Pclass", "Cabin", "Title", "SibSp", "Parch", "Embarked", "Sex"]

cat_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))



preprocessor = ColumnTransformer(transformers=[

    ("num", num_pipeline, num_features),

    ("cat", cat_pipeline, cat_features)

])



model = RandomForestClassifier()
#Train data



Xpreprocessed = data_preparation(train)



features = ["Pclass", "Sex", "SibSp", "Parch", "Cabin", "Title", "Embarked"]



y_train = Xpreprocessed.Survived

X_train = Xpreprocessed.drop(["Survived"], axis = 1)





#X_train, X_test, y_train, y_test = train_test_split(Xpreprocessed,Xpreprocessed.Survived,test_size=100, random_state = 50)

#X_test = X_test.drop(["Survived"], axis = 1)

#X_train = X_train.drop(["Survived"], axis = 1)

X_train = pd.get_dummies(X_train[features])

#X_test = pd.get_dummies(X_test[features])
param_grid = {

    'bootstrap': [True],

    'max_depth': [90, 100, 110, 120],

    'max_features': [2, 3, 4, 5],

    'min_samples_leaf': [2, 3, 4, 5],

    'min_samples_split': [6, 7, 8, 10, 12],

    'n_estimators': [100, 200, 300, 1000]

}
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 

                          cv = 4, n_jobs = -1, verbose = 2)



grid_search.fit(X_train, y_train)



grid_search.best_params_
X_test = data_preparation(test)

X_train = data_preparation(train)

y_train = X_train.Survived

X_train = X_train.drop(["Survived"], axis = 1)



#model = RandomForestClassifier(bootstrap = True,

#                              max_depth = 90,

#                              max_features = 5,

#                              min_samples_leaf = 2,

#                              min_samples_split = 6,

#                              n_estimators = 100)



#classification_process = make_pipeline(preprocessor, model)

#classification_process.fit(X_train, y_train)
param_grid = {

    'bootstrap': [True],

    'max_depth': [90, 100, 110],

    'max_features': [4, 5, 6, 7],

    'min_samples_leaf': [1, 2, 3, 4],

    'min_samples_split': [5, 6, 7],

    'n_estimators': [75, 100, 125]

}
y_train = Xpreprocessed.Survived

A = Xpreprocessed.Age

F = Xpreprocessed.Fare

X_train = Xpreprocessed.drop(["Survived"], axis = 1)

X_train = pd.get_dummies(X_train[features])

X_train["Age"] = A

X_train["Fare"] = F
model = RandomForestClassifier()

grid_search = GridSearchCV(model, param_grid, 

                          cv = 4, n_jobs = -1, verbose = 2)



grid_search.fit(X_train, y_train)



grid_search.best_params_
X_test = data_preparation(test)

A = X_test.Age

F = X_test.Fare

X_test = pd.get_dummies(X_test[features])

X_test["Age"] = A

X_test["Fare"] = F



predictions = grid_search.predict(X_test)

result = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

result.to_csv('aftergrid.csv', index=False)

print(result)