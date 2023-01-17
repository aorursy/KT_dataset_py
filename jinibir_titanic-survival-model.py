import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
plt.rc("font", size=14)

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error

import statsmodels.api as sm
df_titanic_train = pd.read_csv('../input/train.csv')
df_titanic_test = pd.read_csv('../input/test.csv')
df_titanic_train.info()
df_titanic_train.describe()
# get the number of missing data points per column
missing_values_count = df_titanic_train.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:12]
# get the number of missing data points per column
missing_values_count2 = df_titanic_test.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count2[0:11]
df_titanic_train = df_titanic_train.drop(columns = ['Cabin'])
df_titanic_train.head()
df_titanic_test = df_titanic_test.drop(columns = ['Cabin'])
df_titanic_test.head()
df_titanic_train["Embarked"] = df_titanic_train["Embarked"].fillna("Unknown")
df_titanic_train.head()
train_mean = df_titanic_train['Age'].mean()
df_titanic_train['Age'] = df_titanic_train['Age'].fillna(train_mean)
train_mean2 = df_titanic_test['Age'].mean()
df_titanic_test['Age'] = df_titanic_test['Age'].fillna(train_mean2)
df_titanic_test["Fare"] = df_titanic_test["Fare"].fillna(0)
df_titanic_test.head()
df_titanic_train = df_titanic_train.drop(columns = ['PassengerId','Ticket'])
df_titanic_test = df_titanic_test.drop(columns = ['PassengerId','Ticket'])
# This graph will display the count of the survivors
df_titanic_train['Survived'].value_counts()
sns.countplot(x = 'Survived', data = df_titanic_train, palette = 'husl')
plt.show()
plt.savefig('survive_count')
# Survived vs Gender Class

%matplotlib inline
pd.crosstab(df_titanic_train.Sex,df_titanic_train.Survived).plot(kind='bar')
plt.title('Survived vs Gender')
plt.xlabel('Sex')
plt.ylabel('Survived')
plt.savefig('sex_x_survived')
# Survived vs SibSp

%matplotlib inline
pd.crosstab(df_titanic_train.SibSp,df_titanic_train.Survived).plot(kind='bar')
plt.title('Survived vs SibSp')
plt.xlabel('SibSp')
plt.ylabel('Survived')
plt.savefig('sibsp_x_survived')
# Survived vs Parch

%matplotlib inline
pd.crosstab(df_titanic_train.Parch,df_titanic_train.Survived).plot(kind='bar')
plt.title('Survived vs Parch')
plt.xlabel('Parch')
plt.ylabel('Survived')
plt.savefig('parch_x_survived')
# Age Distribution

df_titanic_train.Age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Survived')
plt.savefig('hist_age')
df_titanic_train['Sex_Male'] = pd.np.where(df_titanic_train.Sex.str.contains("male"),0,1)
df_titanic_train['Sex_Female'] = pd.np.where(df_titanic_train.Sex.str.contains("female"),0,1)
df_titanic_train['Embarked_S'] = pd.np.where(df_titanic_train.Sex.str.contains("S"),0,1)
df_titanic_train['Embarked_C'] = pd.np.where(df_titanic_train.Sex.str.contains("C"),0,1)
df_titanic_train['Embarked_Q'] = pd.np.where(df_titanic_train.Sex.str.contains("Q"),0,1)

df_titanic_train.head()
df_titanic_train = df_titanic_train.drop(columns = ['Sex','Embarked'])
df_titanic_test['Sex_Male'] = pd.np.where(df_titanic_test.Sex.str.contains("male"),0,1)
df_titanic_test['Sex_Female'] = pd.np.where(df_titanic_test.Sex.str.contains("female"),0,1)
df_titanic_test['Embarked_S'] = pd.np.where(df_titanic_test.Sex.str.contains("S"),0,1)
df_titanic_test['Embarked_C'] = pd.np.where(df_titanic_test.Sex.str.contains("C"),0,1)
df_titanic_test['Embarked_Q'] = pd.np.where(df_titanic_test.Sex.str.contains("Q"),0,1)

df_titanic_test.head()
df_titanic_test = df_titanic_test.drop(columns = ['Sex','Embarked'])
y_train = df_titanic_train['Survived']
train_predictors = ['Pclass','Age','SibSp','Parch','Fare','Sex_Male','Sex_Female','Embarked_S','Embarked_C','Embarked_Q']
X_train = df_titanic_train[train_predictors]
test_predictors = ['Pclass','Age','SibSp','Parch','Fare','Sex_Male','Sex_Female','Embarked_S','Embarked_C','Embarked_Q']
X_test = df_titanic_test[test_predictors]
forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
survive_predict = forest_model.predict(X_test)
forest_model.score(X_train,y_train)
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
logreg_pred = logreg_model.predict(X_test)
print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(logreg_model.score(X_train, y_train)))
#dtree_model = DecisionTreeRegressor()
#dtree_model.fit(X_train, y_train)
#survive_predict = forest_model.predict(X_test)
#dtree_model.score(X_train,y_train)
submission = pd.DataFrame({
        "Name": df_titanic_test["Name"],
        "Survived": survive_predict
    })
submission.to_csv('titanic.csv', index=False)
