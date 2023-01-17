## We will start by importing all the necessary libraries

## For Analysis
import numpy as np
import pandas as pd

## For visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

## For prediction
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier 



## After all our libraries are imported, its time to import the dataset
## We will use the pandas read_csv function to import the test.csv and the train.csv

train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
## As the standard norm, we will have a look at the first few rows
train_df.head()
## We will drop the PassengerId and Ticket columns since those two columns are not required for analysis
train_df.drop(['PassengerId','Ticket'],axis = 1,inplace = True)
## Viewing the dataset one more time
train_df.head()
## Embarked, Sex and Pclass are categorical that we will encode later.
train_df.info()
## OBSERVATION
## Cabin has 204 not null values
## Age has 714 not null values
## Embarked has 889 not null values
## We will do data cleaning/Imputation on these columns
train_df.describe()
## OBSERVATION
## Out of 891 passengers on board:
## 25% aged 20 or below,50% aged 28 or below and 75% aged 38 or below. We also had passenger/s whose age was 80
## 50% of passengers travelled with no siblings or spouce while 75% passengers travelled with no parents or children
## 75% of the passengers bought tickets that cost 31 USD or less. However the ticket mean price is 32 USD, so 25% of the passengers
## got expensive tickets
## Drop the Cabin column
train_df.drop(['Cabin'],axis = 1, inplace = True)
## Drop the Name column
train_df.drop(['Name'],axis = 1, inplace = True)
## Drop the fare column since its correlated to Pclass
train_df.drop(['Fare'], axis = 1, inplace = True)
## Check passengder details for all those passengers whose age is missing
## .loc[] -> Access a group of rows and columns by label(s) or a boolean array.
## .loc[] is primarily label based, but may also be used with a boolean array.

train_df.loc[train_df['Age'].isnull()]

## So we will impute all the null values in Age with the median age
train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
## Check passengder details for all those passengers whose Embarkment info is missing
train_df.loc[train_df['Embarked'].isnull()]

## So we will impute all the null values in Embarkment with the mode value
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)
## Checking the data data once more after the data cleaning
train_df.isnull().sum()

## So now our dataset looks as follows
train_df.head(15)
## Our Observations:
## The following are categorial features - Pclass,Sex,SibSp,Parch,Embarked
## The following features are continious - Age
## Observations by Pclass
print('The total number of people survivied from each class\n')
print(train_df[['Pclass','Survived']].groupby(['Pclass'], as_index = False).sum())
print('*'* 50)
print('The average number of people survivied from each class\n')
print(train_df[['Pclass','Survived']].groupby(['Pclass'], as_index = False).mean())

## So the column Pclass has a strong predictive power.
sns.barplot(x="Pclass", y="Survived", data=train_df)
## Observations by Sex
print('The total number of people survived according to gender\n')
print(train_df[['Sex','Survived']].groupby(['Sex']).sum())
print('*'* 50)
print('The average number of people survivied according to gender\n')
print(train_df[['Sex','Survived']].groupby(['Sex']).mean())

## So the column Sex has a strong predictive power.
sns.barplot(x="Sex", y="Survived", data=train_df)
## Observations by SibSp
print('The total number of people survived according to the number of Siblings or Spouce they had\n')
print(train_df[['SibSp','Survived']].groupby(['SibSp']).sum())
print('*'* 50)
print('The average number of people survivied according to the number of Siblings or Spouce they had\n')
print(train_df[["SibSp", "Survived"]].groupby(['SibSp']).mean())

sns.barplot(x="SibSp", y="Survived", data=train_df)
## Observations by Parch
print('The total number of people survived according to the number of Parents or children they had\n')
print(train_df[['Parch','Survived']].groupby(['Parch']).sum())
print('*'* 50)
print('The average number of people survivied according to the number of Parents or children they had\n')
print(train_df[["Parch", "Survived"]].groupby(['Parch']).mean())
sns.barplot(x="Parch", y="Survived", data=train_df)
sns.barplot(x="Embarked", y="Survived",hue="Sex", data=train_df)

## Since Age is a continious numerical feature, a groupby will not make much sense.
## Instead we will plot a barplot after binning

train_df['age_by_decade'] = pd.cut(x=train_df['Age'], bins=[0,10, 20, 30, 40, 50, 60, 80], labels=['Infants','Teenagers','20s', '30s','40s','50s','Oldies'])
## Adding a New column Relatives
## train_df['Relatives'] = train_df['SibSp'] + train_df['Parch']
train_df.head(15)
## Observations by Age groups/bins
print('The total number of people survived according to the Age group\n')
print(train_df[['age_by_decade','Survived']].groupby(['age_by_decade']).sum())
print('*'* 50)
print('The average number of people survivied according to the Age group\n')
print(train_df[["age_by_decade", "Survived"]].groupby(['age_by_decade']).mean())



sns.barplot(x="age_by_decade", y="Survived", data=train_df)
#map each Age value to a numerical value
age_mapping = {'Infants': 1, 'Teenagers': 2, '20s': 3, '30s': 4, '40s': 5, '50s': 6, 'Oldies': 7}
train_df['age_by_decade'] = train_df['age_by_decade'].map(age_mapping)
train_df['age_by_decade'] = pd.to_numeric(train_df['age_by_decade'])


#dropping the Age feature for now, might change
train_df = train_df.drop(['Age'], axis = 1)
## test = test.drop(['Age'], axis = 1)

train_df.head()
#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train_df['Sex'] = train_df['Sex'].map(sex_mapping)
## test['Sex'] = test['Sex'].map(sex_mapping)

train_df.head(15)
LikelySurvived = pd.Series([])
for i in range(len(train_df)):
    if  (train_df['Sex'][i]  == 1) | (train_df['age_by_decade'][i] == 1):
        a = pd.Series([1])
        LikelySurvived = LikelySurvived.append(a)
    else:
        a = pd.Series([0])
        LikelySurvived = LikelySurvived.append(a)
        

train_df['LikelySurvived'] = LikelySurvived.values


LikelyDied = pd.Series([])
for i in range(len(train_df)):
    if  ((train_df['Sex'][i]  == 0) &  (train_df['Pclass'][i]  == 3)) | (train_df['Sex'][i]  == 0  & (train_df['Embarked'][i]  == 'Q')) :
        a = pd.Series([1])
        LikelyDied = LikelyDied.append(a)
    else:
        a = pd.Series([0])
        LikelyDied = LikelyDied.append(a)
        

train_df['LikelyDied'] = LikelyDied.values
#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train_df['Embarked'] = train_df['Embarked'].map(embarked_mapping)
## test['Embarked'] = test['Embarked'].map(embarked_mapping)

train_df.head()
## Checking the correlation between the features
pd.DataFrame(abs(train_df.corr()['Survived']).sort_values(ascending = False))
## The correlation Heatmap
sns.heatmap(train_df.corr(),linewidth = .5)
X = train_df.drop(['Survived'], axis=1)
y = train_df["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.22, random_state = 5)
## Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)
# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)
# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC(max_iter=1200000)
linear_svc.fit(X_train, y_train)
y_pred = linear_svc.predict(X_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)
decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train, y_train)
y_pred = decisiontree.predict(X_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)
## Random Forest
randomforest = RandomForestClassifier()
randomforest.fit(X_train, y_train)
y_pred = randomforest.predict(X_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)
# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
## Stochaistic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)
## Gradient Boosting

gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
y_pred = gbk.predict(X_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)
## XGBoost Classifier
xgb = XGBClassifier(random_state=5,learning_rate=0.01)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_val)
acc_xgb = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_xgb)
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier','XGBoost'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk,acc_xgb]})
models.sort_values(by='Score', ascending=False)
test_df.head(15)
## Dropping columns Cabin, Fare,Ticket,Name
test_df.drop(['Name','Ticket','Fare','Cabin'], axis = 1, inplace = True)

## Imputing missing values
test_df['Age'].fillna(test_df['Age'].median(), inplace = True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace = True)
## Changing colums Sex,Age and Embarked

## Sex
sex_mapping = {"male": 0, "female": 1}
test_df['Sex'] = test_df['Sex'].map(sex_mapping)

## Age
test_df['age_by_decade'] = pd.cut(x=test_df['Age'], bins=[0,10, 20, 30, 40, 50, 60, 80], labels=['Infants','Teenagers','20s', '30s','40s','50s','Oldies'])
age_mapping = {'Infants': 1, 'Teenagers': 2, '20s': 3, '30s': 4, '40s': 5, '50s': 6, 'Oldies': 7}
test_df['age_by_decade'] = test_df['age_by_decade'].map(age_mapping)
test_df['age_by_decade'] = pd.to_numeric(train_df['age_by_decade'])


#dropping the Age feature for now, might change
test_df = test_df.drop(['Age'], axis = 1)

## Embarked
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
test_df['Embarked'] = test_df['Embarked'].map(embarked_mapping)

## test_df['Relatives'] = test_df['SibSp'] + test_df['Parch']

## Adding column LikelySurvive
LikelySurvived = pd.Series([])
for i in range(len(test_df)):
    if  (test_df['Sex'][i]  == 1) | (test_df['age_by_decade'][i] == 1):
        a = pd.Series([1])
        LikelySurvived = LikelySurvived.append(a)
    else:
        a = pd.Series([0])
        LikelySurvived = LikelySurvived.append(a)
        

test_df['LikelySurvived'] = LikelySurvived.values

## Likely Died
LikelyDied = pd.Series([])
for i in range(len(test_df)):
    if  (test_df['Sex'][i]  == 0  &  (test_df['Pclass'][i]  == 3)) | (train_df['Sex'][i]  == 0  & (train_df['Embarked'][i]  == 'Q')) :
        a = pd.Series([1])
        LikelyDied = LikelyDied.append(a)
    else:
        a = pd.Series([0])
        LikelyDied = LikelyDied.append(a)
        

test_df['LikelyDied'] = LikelyDied.values
test_df.head(15)
#set ids as PassengerId and predict survival 
ids = test_df['PassengerId']
predictions = svc.predict(test_df.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)
