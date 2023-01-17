import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.model_selection import train_test_split

%matplotlib inline
titanic_df= pd.read_csv('../input/titanic_data.csv')
titanic_df.info()
titanic_df.head()
titanic_df.describe()
titanic_df.describe(include=['O'])
titanic_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
titanic_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
titanic_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).count()
titanic_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).sum()
sns.countplot(x='Survived',data=titanic_df)
titanic_df['Survived'].value_counts()
titanic_df['Sex'].value_counts()
sns.countplot(x='Survived',data=titanic_df,hue='Sex')
titanic_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
titanic_df['Pclass'].value_counts()
sns.countplot(x='Survived', data=titanic_df, hue='Pclass')
titanic_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
sns.distplot(titanic_df['Age'].dropna(), kde=False, bins=30)
sns.countplot(x='SibSp',data=titanic_df)
titanic_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()
sns.countplot(x='Embarked',data=titanic_df)
titanic_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
plt.figure(figsize=(20,7))

sns.distplot(titanic_df['Fare'],kde=False, bins=30)
sns.boxplot(x='Pclass',y='Age',data=titanic_df)
titanic_df[['Pclass', 'Age']].groupby(['Pclass'], as_index=False).mean()
#CLEANING THE DATA
sns.heatmap(titanic_df.isnull(),yticklabels=False,cbar=False)
def input_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass ==1:

            return 38

        elif Pclass ==2:

            return 30

        else:

            return 25

    else:

        return Age
titanic_df['Age'] = titanic_df[['Age','Pclass']].apply(input_age,axis=1)
titanic_df[['Pclass', 'Age']].groupby(['Pclass'], as_index=False).mean()
titanic_df['Embarked'].fillna('S',inplace=True)
sns.heatmap(titanic_df.isnull(),yticklabels=False,cbar=False)
#Create Dummies
sex = pd.get_dummies(titanic_df['Sex'], drop_first=True)

embark = pd.get_dummies(titanic_df['Embarked'],drop_first=True)
titanic_df = pd.concat([titanic_df,sex,embark],axis=1)
titanic_df.head()
#titanic_df.drop(['PassengerId','Sex','Embarked','Fare','Age','Name','Cabin','Ticket'],axis=1,inplace=True)
titanic_df.drop(['Sex','Embarked','Fare','Age','Name','Cabin','Ticket'],axis=1,inplace=True)
titanic_df.head()
type(titanic_df)
X_train, X_test, y_train, y_test = train_test_split(titanic_df.drop('Survived',axis=1), 

                                                    titanic_df['Survived'], test_size=0.30, 

                                                    random_state=101)
X_train.shape, y_train.shape, X_test.shape
logmodel = LogisticRegression()
logmodel.fit(X_train.drop('PassengerId',axis=1), y_train)
Predictions_log = logmodel.predict(X_test.drop('PassengerId',axis=1))
Predictions_log
logmodel.coef_
print(classification_report(y_test,Predictions_log))
print(confusion_matrix(y_test,Predictions_log))
# get Correlation Coefficient for each feature using Logistic Regression

coeff_df = pd.DataFrame(titanic_df.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logmodel.coef_[0])



# preview

coeff_df
logmodel.intercept_
#Logistic Score

logmodel.score(X_train.drop('PassengerId',axis=1), y_train)
submission_log = pd.DataFrame({

        "PassengerId": X_test['PassengerId'],

        "Survived": Predictions_log

    })
submission_log.head()
submission_log.to_csv('titanic_log.csv',index=False)
#RANDOM FOREST
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train.drop('PassengerId',axis=1), y_train)
Predictions_random_forest = random_forest.predict(X_test.drop('PassengerId',axis=1))
Predictions_random_forest
Predictions_random_forest.shape
print(classification_report(y_test,Predictions_random_forest))
print(confusion_matrix(y_test,Predictions_random_forest))
random_forest.score(X_train.drop('PassengerId',axis=1), y_train)
submission_random_forest = pd.DataFrame({

        "PassengerId": X_test['PassengerId'],

        "Survived": Predictions_random_forest

    })
submission_random_forest.to_csv('titanic_rf.csv',index=False)