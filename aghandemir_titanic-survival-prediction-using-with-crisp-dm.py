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
#take a look at the training data's head and tail

head = train.head()

sample= train.sample(5)

tail = train.tail()

data_row = pd.concat([head,sample,tail], axis =0, ignore_index =True)

data_row
## get simple statistics on this dataset

train.describe(include="all").T
#get information about the dataset

train.info
#get a list of the features within the dataset

print(train.columns)
#check for any other unusable values

train.isnull().any()
print(pd.isnull(train).sum())
!pip install missingno
import missingno as msno
msno.bar(train);
# Classes of some categorical variables
def bar_plot(variable):

    """

    input: Variable ex:Sex

    output: barplot&value count

    """

    #get feature

    var=train[variable]

    #Count of Categorical variable(value/sample) 

    varValue=var.value_counts()

    

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))
category1=["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]

for c in category1:

    bar_plot(c)
train['Ticket'].value_counts()
train['Cabin'].value_counts()
# Classes of some numerical variables
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train[variable], bins=50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title(" {} distribution with hist".format(variable))

    plt.show()
numericVar=["Fare", "Age", "PassengerId"]

for n in numericVar:

    plot_hist(n)
#draw a bar plot of survival by Sex



sns.barplot(x="Sex", y="Survived", data=train)

#Print percentages of females vs. males that survive

print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived  :", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)

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
#sort the ages into logical categories



train["Age"] = train["Age"].fillna(train["Age"].median())

test["Age"] = test["Age"].fillna(test["Age"].median())

bins = [-1, 0, 5, 12, 18, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



#draw a bar plot of AgeGroup vs. survival



sns.barplot(x="AgeGroup", y="Survived", data=train)

plt.show()
#Create CabinBool variable which states if someone has a Cabin data or not:



train["CabinBool"] = (train["Cabin"].notnull().astype('int'))

test["CabinBool"] = (test["Cabin"].notnull().astype('int'))



#calculate percentages of CabinBool vs. survived



print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)



#draw a bar plot of CabinBool vs. survival



sns.barplot(x="CabinBool", y="Survived", data=train)

plt.show()

test.describe()
test.head()
train.head()
# We can drop the Ticket, Name and Cabin for the test and train data



test = test.drop(['Ticket'], axis = 1)

test = test.drop(['Cabin'], axis = 1)

test = test.drop(['Name'], axis = 1)

test.head()



 
train = train.drop(['Ticket'], axis = 1)

train = train.drop(['Cabin'], axis = 1)

train = train.drop(['Name'], axis = 1)

train.head()
train.isnull().sum()
test.isnull().sum()
train.describe().T

test.describe().T
# It looks like there is a problem in Fare max data. Visualize with boxplot.



sns.boxplot(x = train['Fare']);
Q1 = train['Fare'].quantile(0.25)

Q3 = train['Fare'].quantile(0.75)

IQR = Q3 - Q1



lower_limit = Q1- 1.5*IQR

lower_limit



upper_limit = Q3 + 1.5*IQR

upper_limit
# observations with Fare data higher than the upper limit:



train['Fare'] > (upper_limit)
train.sort_values("Fare", ascending=False).head()
# In boxplot, there are too many data higher than upper limit; we can not change all. Just repress the highest value -512- 



train['Fare'] = train['Fare'].replace(512.3292, 300)
train.sort_values("Fare", ascending=False).head()
test.sort_values("Fare", ascending=False)
test['Fare'] = test['Fare'].replace(512.3292, 300)
test.sort_values("Fare", ascending=False)
train.isnull().sum()
train['Embarked'].value_counts()
train["Embarked"] = train["Embarked"].fillna("S")
train.isnull().sum()
test.isnull().sum()
test[test["Fare"].isnull()]
test[["Pclass","Fare"]].groupby("Pclass").mean()
test["Fare"] = test["Fare"].fillna(12)
test.isnull().sum()
train.head(5)
#map each Sex value to a numerical value



sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

train.head()
# Convert Sex values into 1-0:



from sklearn import preprocessing

lbe = preprocessing.LabelEncoder()

test["Sex"] = lbe.fit_transform(test["Sex"])

test.head()
#map each Embarked value to a numerical value



embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)



train.head()
train.head()
# Map each Age value to a numerical value:

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3,  'Young Adult': 4, 'Adult': 5, 'Senior': 6}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
test.head()
#dropping the Age feature for now, might change:

train = train.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)
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
train
test
#Spliting the train data



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

predictors = train.drop(['Survived','PassengerId'], axis=1)

target = train["Survived"]

x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.25, random_state = 0)



x_train.shape

x_test.shape
#Logistic Regression



from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 1)

print(acc_logreg)



#Random Forest



from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_test)

acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 1)

print(acc_randomforest)

#Gradient Boosting Classifier



from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_test)

acc_gbk = round(accuracy_score(y_pred, y_test)*100, 1)

print(acc_gbk)
#Model Tuning



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

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 1)

print(acc_gbk)
test
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = xgb_tuned.predict(test.drop('PassengerId', axis=1))
#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)
output.head()