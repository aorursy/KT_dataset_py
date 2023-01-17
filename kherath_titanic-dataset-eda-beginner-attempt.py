import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

%matplotlib inline

data = pd.read_csv('../input/train.csv')

data.head(10)
data.tail(10)

sns.pairplot(data,x_vars=['Age','Fare','Pclass'],y_vars='Survived',kind='reg',size=7)



data.columns
data.shape

data.isna().sum()




data.groupby('Age').Survived.value_counts(dropna=False)
data.Age.describe()
data.Age.agg(['min','max','mean','std'])
data.Age.agg(['min','max','mean','std']).plot(kind = 'barh')

age_survival = data.loc[data.Survived == 1,'Age'].value_counts().sort_index().plot(figsize=(13,8))

age_survival.set_xlabel('Age')
age_survival.set_ylabel('Survival')

data.loc[(data['Survived']==1) & (data['Sex']=='female') & (data['Age'])]
sns.boxplot(x =data.Sex =='female',y=data['Survived'])
data.loc[data.Sex=='female','Survived'].value_counts()


data.loc[(data['Survived']==1) & (data['Sex']=='female') & (data['Age'])].mean()

sns.pairplot(data,x_vars='Age',y_vars='Survived',kind='reg',size=10)


pd.crosstab(data.Survived,data.Embarked).plot(kind = 'bar')

data.Embarked.value_counts(dropna=False)
data['Embarked'] = data.Embarked.map({'S':0,'C':1,'Q':2})
data.Embarked.value_counts()
data['Embarked'] = data.Embarked.fillna(value = 0.0)

data.Embarked.value_counts(dropna=False)
data.Embarked.shape

data.head()
data['Embarked'].head()
data.Embarked.shape
data.Embarked.isna().sum()


sns.pairplot(data,x_vars='Embarked',y_vars='Survived',kind='reg',size=10)


data.Cabin.value_counts().head()

data[(data.Survived ==1) & (data.Cabin)]



data.loc[data['Age'] <= 15,'Age'] = 0 

data.loc[(data['Age'] > 15) & (data['Age'] <= 30), 'Age'] = 1

data.loc[(data['Age'] > 30) & (data['Age'] <= 50), 'Age'] = 2

data.loc[(data['Age'] > 50),'Age'] = 3

data.Age.head()

data.Age.isna().sum()

data.dropna(subset=['Age'],axis ='index',how='all',inplace=True)
data.Age.isna().sum()
data.Age.value_counts(dropna=False)
data.Age.isna().sum()


sns.pairplot(data,x_vars='Age',y_vars='Survived',size=10,kind='reg')

data.head(10)
data.Age.isna().sum()



pd.crosstab(data.Survived,data.Sex).plot(kind='bar')


data['Sex'] = data.Sex.map({'female':0,'male':1})
data.Sex.values

data.head(10)

data.isna().sum()


data.head(100)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import seaborn as sns
feature_cols = ['Pclass','Sex','Age','Embarked','SibSp','Fare']
sns.pairplot(data,x_vars=feature_cols,y_vars = 'Survived',kind = 'reg',size = 4,aspect=0.9)


feature_cols = ['Pclass','Sex','Age','Embarked','SibSp','Fare']

X = data[feature_cols]

y = data.Survived
print(type(X))
print(X.shape)
print(type(y))
print(y.shape)

#Instantiate the Model

linreg = LinearRegression()
# Fit the model 

linreg.fit(X,y)

linreg.intercept_
linreg.coef_
feature_list = list(zip(feature_cols,linreg.coef_))

feature_list




# 10 fold cross validation with all 4 features

linreg = LinearRegression()

score = cross_val_score(linreg,X,y,cv=10,scoring='neg_mean_squared_error')
score

# Make the scores +

msc_sc = -score
print(msc_sc)

# Calculate te RMSE

rmse = np.sqrt(msc_sc)
print(rmse)

## Print the mean of RMSE

print(rmse.mean())


# Load the test set 

test = pd.read_csv('../input/test.csv')
test.head()

test['Sex'] = test.Sex.map({'female':0,'male':1})
test.Sex.value_counts()

test.Embarked.value_counts()
test['Embarked'] = test.Embarked.map({'S':0,'C':1,'Q':2})
test.Embarked.value_counts()

test.loc[test['Age'] <= 15,'Age'] = 0 

test.loc[(test['Age'] > 15) & (test['Age'] <= 30), 'Age'] = 1

test.loc[(test['Age'] > 30) & (test['Age'] <= 50), 'Age'] = 2

test.loc[(test['Age'] > 50),'Age'] = 3

test.isna().sum()
test.Age.dropna(axis='index',how='any',inplace=True)
test['Age'] = test.Age.isna().sum()

test.isna().sum()



feature_cols = ['Pclass','Sex','Age','Embarked','SibSp']

X_test = test[feature_cols]
linreg = LinearRegression()

linreg.fit(X,y)

y_pred = linreg.predict(X)
y_pred[X_test]




from sklearn.linear_model import LogisticRegression
feature_cols = ['Pclass','Sex','Age','Embarked','SibSp','Fare']

X = data[feature_cols]

y = data.Survived
logreg = LogisticRegression()

logreg.fit(X,y)

score = cross_val_score(logreg,X,y,cv=10,scoring='neg_mean_squared_error')

score

mse = - score
## NOw calculate the RMSE

rmse = np.sqrt(mse)
print(rmse)


test_cols = ['Pclass','Sex','Age','Embarked','SibSp','Fare']

X_test = test[test_cols]
X
X_test
X_test.isna().sum()
X_test.dtypes

X_test.Fare.fillna(X_test.Fare.mean(),inplace=True)

X_test.isna().sum()


logreg = LogisticRegression()

logreg.fit(X,y)
y_pred = logreg.predict(X_test)
y_pred

pd.get_option('display.max_rows')
pd.set_option('display.max_rows',None)

X_test.shape
test.PassengerId.shape

# Create a pandas Dataframe

pd.DataFrame({'PasssngerId':test.PassengerId,'Survived':y_pred})

# now save PassengerId columns as the index

pd.DataFrame({'PasssngerId':test.PassengerId,'Survived':y_pred}).set_index('PasssngerId')


# Finally Convert the file to a CSV file 

pd.DataFrame({'PassengerId':test.PassengerId,'Survived':y_pred}).set_index('PassengerId').to_csv('Titaanic log reg2.csv')




