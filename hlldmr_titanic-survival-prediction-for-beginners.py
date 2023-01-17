# data analysis libraries:

import numpy as np

import pandas as pd



# data visualization libraries:

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# to ignore warnings:

import warnings

warnings.filterwarnings('ignore')



# to display all columns:

pd.set_option('display.max_columns', None)



from sklearn.model_selection import train_test_split, GridSearchCV



#import train and test CSV files



train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
# copy data in order to avoid any change in the original:



train = train_data.copy()

test = test_data.copy()
#take a look at the training data



train.describe(include="all")
#get information about the dataset



train.info()
#get a list of the features within the dataset



print(train.columns)

#head 



train.head()

#head 



test.head()

#tail



train.tail()
#see a sample of the dataset to get an idea of the variables



train.sample(5)



#check for any other unusable values



print(pd.isnull(train).sum())

#see a summary of the training dataset



train.describe().T

100*train.isnull().sum()/len(train)
train['Pclass'].value_counts()
train['Sex'].value_counts()
train['SibSp'].value_counts()
train['Parch'].value_counts()
train['Ticket'].value_counts()
train['Cabin'].value_counts()
train['Embarked'].value_counts()
#draw a bar plot of survival by sex



sns.barplot(x="Sex", y="Survived", data=train)



#print percentages of females vs. males that survive



print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)

#draw a bar plot of survival by Pclass



sns.barplot(x="Pclass", y="Survived", data=train)



#print percentage of people by Pclass that survived



print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
#draw a bar plot for SibSp vs. survival



sns.barplot(x="SibSp", y="Survived", data=train)



#I won't be printing individual percent values for all of these.



print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)



print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of SibSp = 3 who survived:", train["Survived"][train["SibSp"] == 3].value_counts(normalize = True)[1]*100)



print("Percentage of SibSp = 4 who survived:", train["Survived"][train["SibSp"] == 4].value_counts(normalize = True)[1]*100)

#draw a bar plot for Parch vs. survival



sns.barplot(x="Parch", y="Survived", data=train)

plt.show()



train["CabinBool"] = (train["Cabin"].notnull().astype('int'))

test["CabinBool"] = (test["Cabin"].notnull().astype('int'))



#calculate percentages of CabinBool vs. survived



print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)



#draw a bar plot of CabinBool vs. survival



sns.barplot(x="CabinBool", y="Survived", data=train)



plt.show()

test.describe().T
# Create CabinBool variable which states if someone has a Cabin data or not:



train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)



train.head()

train.isnull().sum()
test.isnull().sum()
print(pd.isnull(train.CabinBool).sum())

# We can drop the Ticket feature since it is unlikely to have useful information



train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)



train.head()
train.describe().T

# It looks like there is a problem in Fare max data. Visualize with boxplot.



sns.boxplot(x = train['Fare']);
Q1 = train['Fare'].quantile(0.05)

Q3 = train['Fare'].quantile(0.95)

IQR = Q3 - Q1



lower_limit = Q1- 1.5*IQR

lower_limit



upper_limit = Q3 + 1.5*IQR

upper_limit
# observations with Fare data higher than the upper limit:



train['Fare'] > (upper_limit)
train.sort_values("Fare", ascending=False).head()
# In boxplot, there are too many data higher than upper limit; we can not change all. Just repress the highest value -512- 



train['Fare'] = train['Fare'].replace(512.3292, 270)
train.sort_values("Fare", ascending=False).head()
train.sort_values("Fare", ascending=False)
test.sort_values("Fare", ascending=False)
test['Fare'] = test['Fare'].replace(512.3292, 270)
test.sort_values("Fare", ascending=False)
#drop the name feature since it contains no more useful information.



train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
train.describe().T
train.isnull().sum()
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
train.isnull().sum()
test.isnull().sum()
train.describe().T
train.isnull().sum()
test.isnull().sum()
#now we need to fill in the missing values in the Embarked feature



print("Number of people embarking in Southampton (S):")

southampton = train[train["Embarked"] == "S"].shape[0]

print(southampton)



print("Number of people embarking in Cherbourg (C):")

cherbourg = train[train["Embarked"] == "C"].shape[0]

print(cherbourg)



print("Number of people embarking in Queenstown (Q):")

queenstown = train[train["Embarked"] == "Q"].shape[0]

print(queenstown)

train["Embarked"].value_counts()
#replacing the missing values in the Embarked feature with S



train = train.fillna({"Embarked": "S"})
test = test.fillna({"Embarked": "S"})
train.Embarked
train.isnull().sum()
test.isnull().sum()
print(pd.isnull(train.Embarked).sum())

test[test["Fare"].isnull()]
test[["Pclass","Fare"]].groupby("Pclass").mean()
test["Fare"] = test["Fare"].fillna(12)
test["Fare"].isnull().sum()
#check train data



train.head()
#check test data



test.head()
#map each Sex value to a numerical value



sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)



train.head()
#map each Embarked value to a numerical value



from sklearn import preprocessing



lbe = preprocessing.LabelEncoder()

train["Embarked"] = lbe.fit_transform(train["Embarked"])

test["Embarked"] = lbe.fit_transform(test["Embarked"])

train.head()

bins = [0, 5, 12, 18, 24, 35, 60, np.inf]

mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = mylabels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = mylabels)
# Map each Age value to a numerical value:

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
train.head()
#dropping the Age feature for now, might change:

train = train.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)
train.head()
# Map Fare values into groups of numerical values:

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])
# Drop Fare values:

train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)
train.head()
train.head()
train["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1
test["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1
# Create new feature of family size:



train['Single'] = train['FamilySize'].map(lambda s: 1 if s == 1 else 0)

train['SmallFam'] = train['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

train['MedFam'] = train['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

train['LargeFam'] = train['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
train.head()
# Create new feature of family size:



test['Single'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)

test['SmallFam'] = test['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

test['MedFam'] = test['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

test['LargeFam'] = test['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
test.head()

# Convert Title and Embarked into dummy variables:



train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")
train.head()


test = pd.get_dummies(test, columns = ["Embarked"], prefix="Em")

test.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20, random_state = 0)

x_train.shape
x_test.shape
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_logreg)

from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_test)

acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_randomforest)
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_test)

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gbk)



xgb_params = {

        'n_estimators': [200, 500],

        'subsample': [0.6, 1.0],

        'max_depth': [2,5,8],

        'learning_rate': [0.1,0.01,0.02],

        "min_samples_split": [2,5,10]}



xgb = GradientBoostingClassifier()



xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)



xgb_cv_model.fit(x_train, y_train)



xgb_cv_model.best_params_



xgb = GradientBoostingClassifier(learning_rate = xgb_cv_model.best_params_["learning_rate"], 

                    max_depth = xgb_cv_model.best_params_["max_depth"],

                    min_samples_split = xgb_cv_model.best_params_["min_samples_split"],

                    n_estimators = xgb_cv_model.best_params_["n_estimators"],

                    subsample = xgb_cv_model.best_params_["subsample"])

xgb_tuned =  xgb.fit(x_train,y_train)



y_pred = xgb_tuned.predict(x_test)

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gbk)

test



#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = xgb_tuned.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)



output.head()
