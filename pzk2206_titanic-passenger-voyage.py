# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



import missingno as msno

import seaborn as sns



import re

from catboost import CatBoostRegressor

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score

from sklearn import linear_model

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression, Perceptron

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from math import sqrt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
geb = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

submission_test = test.copy()
print("Train dataset shape", train.shape)

print("test dataset shape", test.shape)
train.head()
test.head()
train.info()
total = train.isnull().sum().sort_values(ascending=False)

percentage = train.isnull().sum() / train.isnull().count() * 100

perc = round(percentage,2).sort_values(ascending = False)

missing = pd.concat([total, perc], axis = 1, keys = ["Total","Missing%"])

missing
msno.matrix(train)
msno.matrix(test)
msno.heatmap(train)
msno.bar(train)
plt.figure(figsize = (20,8))

plt.subplot(1,2,1)

sns.distplot(train.loc[train["Survived"] == 0,"Age"])

plt.title("Age distribution of Category 'Not Survived' passangers ")



plt.subplot(1,2,2)

sns.distplot(train.loc[train["Survived"] == 1,"Age"])

plt.title("Age distribution of Category 'Survived'  passangers")
train.loc[train["Embarked"].isnull(),"Embarked"] = 'S'
plt.figure(figsize = (10,8))

sns.countplot(x = "Sex", hue = "Survived", data = train)
plt.figure(figsize = (10,8))

sns.countplot(x = "Embarked", hue = "Survived", data = train)
def getNameTitle(Name):

    [value] = re.findall('[a-zA-Z]+, \w+.', Name)

    title = value.split(', ')[1]

    surname = value.split(', ')[0]

    return title, surname
titles = {"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Rare": 5}



data = [train, test]

for df in data:

    df['title'] = df['Name'].apply(lambda x: getNameTitle(x)[0])

    df['surname'] = df['Name'].apply(lambda x: getNameTitle(x)[1])

    df["title"] = df["title"].replace(['Lady.', 'Countess.','Capt.', 'Col.','Don.', 'Dr.',

                                       'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'], "Rare")

    df['title'] = df['title'].replace('Mlle.', 'Miss.')

    df['title'] = df['title'].replace('Ms.', 'Miss.')

    df['title'] = df['title'].replace('Mme.', 'Mrs.')

    

    df['title'] = df['title'].map(titles)

    df['title'] = df['title'].fillna(0)

    df['title'] = df['title'].astype(int)
# Update the title manually for passanger with title the countess to Mrs.

train.loc[759, "title"] = 3
sns.catplot(x="Pclass", hue="Sex", col="Survived",

                data=train, kind="count",

                height=7, aspect=.7);
sns.catplot(x="Embarked", hue="Sex", col="Survived",

                data=train, kind="count",

                height=7, aspect=.7);
FacetGrid = sns.FacetGrid(train, row='Embarked', size=4.5, aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )

FacetGrid.add_legend();
data = [train, test]

for df in data:

    df["familySize"] = df["SibSp"] + df["Parch"] + 1 

    df['relative'] = df['SibSp'] + df["Parch"]

    df.loc[df["relative"] > 0, 'alone'] = 0

    df.loc[df["relative"] == 0, 'alone'] = 1

    df["alone"] = df["alone"].astype(int)
axes = sns.factorplot('familySize','Survived', 

                      data=train, aspect = 2.5, )
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data = [train, test]



for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int)

train = train.drop(['Cabin'], axis=1)

test = test.drop(['Cabin'], axis=1)
data = [train, test]

sex = {"male": 0, "female": 1}

for df in data:

    df["Sex"] = df["Sex"].map(sex)

    df["Sex"] = df["Sex"].astype(int)

    

Idata = train.drop(["PassengerId","Survived", 'Name',"Ticket","surname"], axis = 1)



df_train = Idata.loc[train["Age"].notnull()]

df_test = Idata.loc[train["Age"].isnull()]



XItrain = df_train.drop('Age', axis = 1)

YItrain = df_train["Age"]

XItest = df_test.drop('Age', axis = 1)

YItest = df_test["Age"]



xtrain, xval, ytrain, yval = train_test_split(XItrain, YItrain, test_size= 0.2, random_state = 42)

categorical_features_indices = np.where(XItrain.dtypes != np.float)[0]

model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')

model.fit(xtrain, ytrain,eval_set = (xval, yval),cat_features=categorical_features_indices,plot=True)



yhat = model.predict(xval)



print("RMSE value:", sqrt(mean_squared_error(yval, yhat)))
cols = XItrain.columns

data = [train, test]

for df in data:

    nullList = df.loc[df["Age"].isnull()].index.tolist()

    for l in nullList:

        yhatValue = model.predict(df.loc[l,cols].values)

        df.loc[l,"Age"] = yhatValue
data = [train, test]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'AgeCat'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'AgeCat'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'AgeCat'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'AgeCat'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'AgeCat'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'AgeCat'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'AgeCat'] = 6

    dataset.loc[ dataset['Age'] > 66, 'AgeCat'] = 6

    dataset['AgeCat'] = dataset['AgeCat'].astype(int)
train["AgeCat"].value_counts()
sns.catplot(x="Sex", hue="AgeCat", col="Survived",

                data=train, kind="count",

                height=7, aspect=.7);
plt.figure(figsize = (10,8))

sns.countplot(x = "familySize", hue = "Survived", data = train)
ports = {"S": 0, "C": 1, "Q": 2}

data = [train, test]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)

    dataset['Fare'] = dataset['Fare'].fillna(0)
dropColumns =  ["PassengerId", 'Name', 'Ticket', 'surname']

train = train.drop(dropColumns, axis = 1)
X_train = train.drop("Survived", axis = 1)

y_train = train["Survived"]

X_test = test.drop(dropColumns, axis = 1)
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, y_train)

yhat = sgd.predict(X_test)



sgd.score(X_train, y_train)



acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)

acc_sgd
random_forest = RandomForestClassifier(n_estimators=100, random_state = 42)

random_forest.fit(X_train, y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
logreg = LogisticRegression()

logreg.fit(X_train, y_train)



Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, y_train) * 100, 2)

acc_log
knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, y_train)  

Y_pred = knn.predict(X_test)  

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

acc_knn
perceptron = Perceptron(max_iter=5)

perceptron.fit(X_train, y_train)



Y_pred = perceptron.predict(X_test)



acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, y_train)  

Y_pred = decision_tree.predict(X_test)  

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

acc_decision_tree
gaussian = GaussianNB() 

gaussian.fit(X_train, y_train)  

Y_pred = gaussian.predict(X_test)  

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

acc_gaussian
results = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 

              'Decision Tree'],

    'Score': [acc_linear_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(9)
rf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(rf, X_train, y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head(15)
train_df  = train.drop("alone", axis=1)

test_df  = test.drop("alone", axis=1)



train_df  = train.drop("Parch", axis=1)

test_df  = test.drop("Parch", axis=1)
# Random Forest



random_forest = RandomForestClassifier(n_estimators=200, 

                                       oob_score = True,

                                       criterion = "gini", 

                                       min_samples_leaf = 2, 

                                       min_samples_split = 10,   

                                       max_features='auto', 

                                       random_state=42, 

                                       n_jobs=-1)

random_forest.fit(X_train, y_train)

Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, y_train)



acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")
test = test.drop(dropColumns, axis = 1)

Y_hat = random_forest.predict(test)
submission = pd.DataFrame({"PassengerId":submission_test["PassengerId"],

        "Survived":pd.Series(Y_hat)})

submission.to_csv("Submission.csv", index = False)