#Importing libraries



import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from math import sqrt



%matplotlib inline
#Loading the datsets

titanic_train = pd.read_csv('../input/train.csv')

titanic_test = pd.read_csv('../input/test.csv')

y_test = pd.read_csv('../input/gender_submission.csv')
titanic_train.head()
#Dropping the string columns that cannot be converted to numeric

#String information not useful in ML models



titanic_train.drop(['PassengerId','Name','Ticket'],inplace=True,axis=1)

titanic_test.drop(['PassengerId','Name','Ticket'],inplace=True,axis=1)

titanic_train.head()
#Checking the distribution of null values

sns.heatmap(titanic_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Lot of data missing from cabin colummn. 

#Cannot devise a good strategy to fill in the missing values

titanic_train.drop('Cabin',axis=1,inplace=True)

titanic_test.drop('Cabin',axis=1,inplace=True)

titanic_train.head()
sns.countplot('Survived',data=titanic_train,palette='plasma',hue='Sex')
sns.countplot('Survived',hue='Pclass',data=titanic_train)
plt.figure(figsize=(8,6))

sns.scatterplot('Age','Fare',data=titanic_train,hue='Survived',palette='viridis')
sns.countplot('Embarked',data=titanic_train,hue='Pclass')
def impute_age(cols):

    Age=cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return titanic_train.groupby(['Pclass']).Age.mean().round(2).iloc[0]

        elif Pclass == 2:

            return titanic_train.groupby(['Pclass']).Age.mean().round(2).iloc[1]

        elif Pclass == 3 :

            return titanic_train.groupby(['Pclass']).Age.mean().round(2).iloc[2]

    else:

        return Age
titanic_train['Age'] = titanic_train[['Age','Pclass']].apply(impute_age,axis=1)

titanic_test['Age'] = titanic_test[['Age','Pclass']].apply(impute_age,axis=1)
def impute_fare(cols):

    Fare = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Fare):

        if Pclass == 1:

            return titanic_test.groupby(['Pclass']).Fare.mean().round(2).iloc[0]

        elif Pclass == 2:

            return titanic_test.groupby(['Pclass']).Fare.mean().round(2).iloc[1]

        elif Pclass == 3:

            return titanic_test.groupby(['Pclass']).Fare.mean().round(2).iloc[2]

    else:

        return Fare

    

titanic_test['Fare'] = titanic_test[['Fare','Pclass']].apply(impute_fare,axis = 1)
titanic_train = pd.get_dummies(columns=['Pclass','Sex','Embarked'],data=titanic_train,drop_first=True)

titanic_test = pd.get_dummies(columns=['Pclass','Sex','Embarked'],data=titanic_test,drop_first=True)

titanic_train.head()#.drop(labels=['Pclass','Sex','Embarked'],axis=1,inplace=True)
titanic_test.head()
X_train = titanic_train.drop('Survived',axis=1).values

y_train =  titanic_train['Survived'].values

X_test = titanic_test.values

y_test = y_test.drop('PassengerId',axis=1).values

y_test = y_test.ravel()
#Scaling the values to avoid overfitting on particular feature due to large range



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
lr = LinearRegression()

lr.fit(X_train,y_train)

pred_lr = lr.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, pred_lr))

rmse