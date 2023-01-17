# data analysis libraries:

import numpy as np

import pandas as pd



# data visualization libraries:

import matplotlib.pyplot as plt

import seaborn as sns



# to ignore warnings:

import warnings

warnings.filterwarnings('ignore')



# to display all columns:

pd.set_option('display.max_columns', None)



from sklearn.model_selection import train_test_split, GridSearchCV
# to import train and test of titanic dataset from kaggle

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
#  to copy titanic datasets ( as train and test) into another variable.

dtr = train_data.copy()

dts = test_data.copy()
dtr.info()
dts.info()
# To combine the dtr and dts data into one variable. 

# (We do this so as not to repeat the same operations on both notebooks (dtr and dts) on both datasets.)

df = pd.concat([dtr,dts], ignore_index=True)
df.head()
df.tail()
# some of the operations on dataset that has numerical varibales. it is meaningless for categorical variables.

df.describe().T
df["Pclass"].value_counts()
df["Sex"].value_counts()
df["Embarked"].value_counts()
df["SibSp"].value_counts()
df["Parch"].value_counts()
df["Ticket"].value_counts()
sns.barplot(x= "Pclass", y = "Survived", data = df);
sns.barplot(x= "Sex", y = "Survived", data = df);
sns.barplot(x= "Embarked", y= "Survived", data =df);
sns.barplot(x= "SibSp", y = "Survived", data= df);
sns.barplot(x= "Parch", y = "Survived", data= df);
df.head()
# To assign categorically 1 to passengers with any cabin number and 0 to those without a cabin number.

df["Cabin"] = df["Cabin"].notnull().astype("int")
df["Cabin"].value_counts()
sns.barplot(x= "Cabin", y = "Survived", data= df);
df["Embarked"].value_counts()
df["Embarked"] = df["Embarked"].fillna("S")
df.info()
df["Age"] = df["Age"].fillna(df["Age"].mean())
df.isnull().sum()
df[df["Fare"].isnull()]


df[["Pclass", "Fare"]].groupby("Pclass").mean()
df["Fare"][1043] = 13
df.isnull().sum()
df.head()
df.describe().T
sns.boxplot(x=df["Fare"])
Q1 = df["Fare"].quantile(0.25)

Q1
Q3 =  df["Fare"].quantile(0.75)

Q3
IQR = Q3-Q1
low_limit = Q1 - 1.5*IQR

high_limit = Q3 + 1.5*IQR

high_limit
df[df["Fare"] > high_limit]
df["Fare"].sort_values(ascending=False).head()
df["Fare"] = df["Fare"].replace(512.3292, 263)
df.info()
df.head()
embarked_mapping = {"S":1, "C":2, "Q": 3} 

df["Embarked"] = df["Embarked"].map(embarked_mapping)



#2. method



# for i in range(0, len(df["Embarked"])):

#     if df["Embarked"][i] == "S":

#         df["Embarked"][i] = 1

#     elif df["Embarked"][i] == "C":

#         df["Embarked"][i] = 2

#     elif df["Embarked"][i] == "Q":

#         df["Embarked"][i] = 3
df.head(20)
df.drop(["Ticket"],axis =1, inplace = True)
df.head()
for i in range(0, len(df["Sex"])):

    if df["Sex"][i] == "male":

        df["Sex"][i] = 1

    elif df["Sex"][i] == "female":

        df["Sex"][i] = 0

# 2. method

# # from sklearn import preprocessing

# # lbe = preprocessing.LabelEncoder()

# # df.Sex = lbe.fit_transform(df.Sex)

   

df.head()
df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
df.head()
df["Title"].value_counts()
df.Title = df['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

df.Title = df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

df.Title = df['Title'].replace('Mlle', 'Miss')

df.Title = df['Title'].replace('Ms', 'Miss')

df.Title = df['Title'].replace('Mme', 'Mrs')
df.Title.value_counts()
df[["Title", "Survived"]].groupby(["Title"], as_index = False ).mean()

Title_mapping = {"Mr":1, "Miss":2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5} 

df["Title"] = df["Title"].map(Title_mapping)
df.head()
df.drop("Name", axis =1, inplace= True)
df.head()


#  to make an Agegroup

# df["AgeGroup"] = 0

# for i in range(0, len(df["Age"])):

#     if df["Age"][i] <= 5:

#         df["AgeGroup"][i] = 1

#     elif df["Age"][i] <= 12:

#         df["AgeGroup"][i] = 2

#     elif df["Age"][i] <= 18:

#         df["AgeGroup"][i] = 3

#     elif df["Age"][i] <= 24:

#         df["AgeGroup"][i] = 4

#     elif df["Age"][i] <= 35:

#         df["AgeGroup"][i] = 5

#     elif df["Age"][i] <= 60:

#         df["AgeGroup"][i] = 6

#     elif df["Age"][i] > 60:

#         df["AgeGroup"][i] = 7

# 2. Method

# bins = [0, 5, 12, 18, 24, 35, 60, np.inf]

# mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

# df['AgeGroup'] = pd.cut(df["Age"], bins, labels = mylabels)



# age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

# df['AgeGroup'] = df['AgeGroup'].map(age_mapping)



# df.drop("Age", axis = 1, inplace = True)
#  to make Fares in Group



# df["FareBand"]= pd.qcut(df["Fare"], 5 , [1,2,3,4,5])

# df.FareBand.value_counts()

# df.drop("Fare", axis= 1, inplace = True)
df["FamilySize"] = df["SibSp"] + df["Parch"]+1
df.head()
df["Single"] = df["FamilySize"].map(lambda x: 1 if x ==   1 else 0)

df["SmaFam"] = df["FamilySize"].map(lambda x: 1 if x ==   2 else 0)

df["MedFam"] = df["FamilySize"].map(lambda x: 1 if 3<=x<= 4 else 0)

df["LarFam"] = df["FamilySize"].map(lambda x: 1 if x >    4 else 0)
df.head(5)
df = pd.get_dummies( df, columns = ["Title"], prefix = "Tit")

df = pd.get_dummies( df, columns = ["Embarked"], prefix = "Em")
df.head()
df.info()
# Applying "One Hot Encoding" method in "Pclass" Variable

# df["Pclass"] = df["Pclass"].astype("category")

# df = pd.get_dummies(df, columns = ["Pclass"], prefix = "Pc")

# df.head()
df.drop(["Parch","SibSp"], axis =1, inplace =True)
df.info()
df.head()
dtr = df[0:891]
dtr.info()
dtr.head()
dtr["Survived"] = dtr["Survived"].astype("int")
dtr.head()
dts = df[891:]

dts.index = dts.index -891

dts.head()
dts.info()
dts.drop("Survived", axis =1, inplace =True)

dts.head(5)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

predictors = dtr.drop(["Survived","PassengerId"], axis = 1)

target = dtr["Survived"]

x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20, random_state = 42)
x_train.shape
x_test.shape
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_logreg)
dts.head(5)
# Random  Forest Classifier Maschine Learning Model



from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_test)

acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_randomforest)
#set ids as PassengerId and predict survival 

ids = dts['PassengerId']

predictions = logreg.predict(dts.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv',index=False)
output.head()
# GradientBoosting Classifier Maschine Learning Model



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