# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Importing training
df_train=pd.read_csv('../input/train.csv')
df_train.head()
# Importing test data
df_test=pd.read_csv('../input/test.csv')
df_test.head()
# Summary of train data
df_train.describe()
df_train.info()
# NULL Heatmap for Train data
sns.heatmap(df_train.isnull(),yticklabels=False, cbar=False,cmap='inferno',annot=True)
sns.countplot(x='Survived',hue='Sex',data=df_train,palette='afmhot')
sns.countplot(x='Survived',hue='Pclass',data=df_train,palette='hsv')
sns.countplot(x='Survived',hue='Parch',data=df_train,palette='rainbow')
sns.countplot(x='Survived',hue='SibSp',data=df_train,palette='rainbow')
sns.countplot(x='Survived',hue='Embarked',data=df_train,palette='rainbow')
sns.boxplot(x='Pclass',y='Age',data=df_train)
# Calculating Mean Age for each Passenger Class
df_train.groupby('Pclass', as_index=False)['Age'].mean()
sns.boxplot(x='Sex',y='Age',data=df_train)
# Calculating mean age across sex distribution
df_train.groupby('Sex', as_index=False)['Age'].mean()
sns.boxplot(x='Embarked',y='Age',data=df_train)
# Since it is evident from the above plots and analysis that Age has well-defined relation with respect to both Sex and Passenger Class.
# Computing mean Age across Sex and Passenger Class - This will be used for imputing the Age. 
df_train.groupby(['Sex','Pclass'])['Age'].mean()
# Age Imputation
def ImputeAge(column):
    Age = column[0]
    Sex = column[1]
    Pclass=column[2]

    if pd.isnull(Age):
        if Sex == 'male' and Pclass==1:
            return 41
        elif Sex == 'male' and Pclass==2:
            return 31
        elif Sex == 'male' and Pclass==3:
            return 26
        elif Sex == 'female' and Pclass==1:
            return 35
        elif Sex == 'female' and Pclass==2:
            return 29
        else:
            return 22
    else:
        return Age
    
df_train['Age'] = df_train[['Age','Sex','Pclass']].apply(ImputeAge,axis=1)
df_test['Age'] = df_test[['Age','Sex','Pclass']].apply(ImputeAge,axis=1)

def ImputeAge(column):
    Age = column[0]
    Sex = column[1]
    Pclass=column[2]

    if pd.isnull(Age):
        if Sex == 'male' and Pclass==1:
            return 41
        elif Sex == 'male' and Pclass==2:
            return 31
        elif Sex == 'male' and Pclass==3:
            return 26
        elif Sex == 'female' and Pclass==1:
            return 35
        elif Sex == 'female' and Pclass==2:
            return 29
        else:
            return 22
    else:
        return Age
    
df_train['Age'] = df_train[['Age','Sex','Pclass']].apply(ImputeAge,axis=1)
df_test['Age'] = df_test[['Age','Sex','Pclass']].apply(ImputeAge,axis=1)
# NULL Heatmap to visualize that Age is correctly imputed.
sns.heatmap(df_train.isnull(),yticklabels=False, cbar=False,cmap='inferno',annot=True)
# Dropping Cabin from both Test and Train data as we do not have enough data across Cabin to predict Survival
df_train.drop('Cabin', axis=1, inplace=True)
df_test.drop('Cabin', axis=1, inplace=True)
# Heatmap of the processed train data
sns.heatmap(df_train.isnull(),yticklabels=False, cbar=False,cmap='inferno',annot=True)
# Logic to replace the NULL Fare in df_test
df_test['Fare'].fillna(df_test['Fare'].mean(), inplace=True) 
df_test.info()
# Converting categorical variables into indicator variables
sex = pd.get_dummies(df_train['Sex'],drop_first=True)
embark = pd.get_dummies(df_train['Embarked'],drop_first=True)
# Name and Ticket have no role in the model prediction
# Replacing Sex and Embarked columns with the new indicator variables
df_train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df_train = pd.concat([df_train,sex,embark],axis=1)
# Repeating the above process for test
sex = pd.get_dummies(df_test['Sex'],drop_first=True)
embark = pd.get_dummies(df_test['Embarked'],drop_first=True)
df_test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df_test = pd.concat([df_test,sex,embark],axis=1)
df_train.drop(['PassengerId'],axis=1,inplace=True)
Passenger_ID = df_test['PassengerId'] # Saving for later
df_test.drop(['PassengerId'],axis=1,inplace=True)
# Fully processed train data
df_train.head()
# Fully processed test data
df_test.head()
# PassengerId
Passenger_ID
# Using Train-Test Split to randomize the data for Predictive modelling
from sklearn.model_selection import train_test_split

x = df_train.drop('Survived', axis = 1)
y = df_train['Survived']

x_train, x_test, y_train, y_test = train_test_split(df_train.drop('Survived',axis=1),df_train['Survived'], test_size = 0.25,random_state=100)
# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train,y_train)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
acc_log
# Decision Tree Classifier
from sklearn import tree

treeclf = tree.DecisionTreeRegressor()
treeclf.fit(x_train,y_train)
acc_tree = round(treeclf.score(x_train, y_train) * 100, 2)
acc_tree
# Random Forest
from sklearn.ensemble import RandomForestClassifier

ranclf =RandomForestClassifier(n_estimators=20, max_depth=None,min_samples_split=2, random_state=0)
ranclf.fit(x_train,y_train)
acc_ranclf = round(ranclf.score(x_train, y_train) * 100, 2)
acc_ranclf
# Predicting Survival values for Test data.
survived=ranclf.predict(df_test)
# Feeding PassengerId and Survived into Test data
df_test['Survived']=survived
df_test['PassengerID']=Passenger_ID
df_test
sns.countplot(x='Survived',hue='male',data=df_train,palette='afmhot')
sns.countplot(x='Survived',hue='male',data=df_test,palette='afmhot')
sns.countplot(x='Survived',hue='Pclass',data=df_train,palette='afmhot')
sns.countplot(x='Survived',hue='Pclass',data=df_test,palette='afmhot')
df_test[['PassengerID', 'Survived']].to_csv('Titanic_LogRegression.csv', index=False)
acc_ranclf = round(ranclf.score(x_train, y_train) * 100, 2)
acc_ranclf