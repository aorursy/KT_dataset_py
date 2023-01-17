import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

from sklearn import datasets, linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
df_train = pd.read_csv('../input/train.csv', index_col='PassengerId')

df_test = pd.read_csv('../input/test.csv', index_col='PassengerId')



y_train = df_train['Survived']
df_test.info()
sns.countplot(df_train['Survived'], palette='hls')
fig, ax = plt.subplots(1,3 , figsize=(10, 6) , sharex='col', sharey='row')

a = sns.countplot(x = 'Sex' , data = df_train , ax = ax[0] , palette='hls')

b = sns.countplot(x = 'Sex' , data = df_train[df_train['Survived'] == 1] , ax = ax[1] , palette='hls')

c = sns.countplot(x = 'Sex' , data = df_train[ ((df_train['Age'] < 21) & (df_train['Survived'] == 1)) ] , palette='hls')



ax[0].set_title('All passenger')

ax[1].set_title('Survived passenger')

ax[2].set_title('Survived passenger under age 21')
fig, ax = plt.subplots(1,3 , figsize=(10, 6) , sharex='col', sharey='row')

a = sns.countplot(x = 'Pclass' , data=df_train , ax = ax[0] , palette='hls')

b = sns.countplot(x = 'Pclass' , data= df_train[df_train['Survived'] == 1] , ax = ax[1] , palette='hls')

c = sns.countplot(x = 'Pclass' , data= df_train[ ((df_train['Age'] < 21) & (df_train['Survived'] == 1)) ] , palette='hls')



ax[0].set_title('All passenger')

ax[1].set_title('Survived passenger')

ax[2].set_title('Survived passenger under age 21')
# check for missing values in the train dataset  

df_train.isna().sum()
# check for missing values in the test dataset

df_test.isna().sum()
df_train.shape
df_train.isna().sum()
df_test.isna().sum()
# Correlation matrix between numerical values

g = sns.heatmap(df_train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
g = sns.heatmap(df_train[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)
df_train['Sex'] = df_train['Sex'].map( {'male':1, 'female':0})

df_test['Sex'] = df_test['Sex'].map( {'male':1, 'female':0})
dummies_embarked = pd.get_dummies(df_train['Embarked'])

df_train = pd.concat([df_train, dummies_embarked], axis=1)



dummies_embarked = pd.get_dummies(df_test['Embarked'])

df_test = pd.concat([df_test, dummies_embarked], axis=1)
df_train.Pclass.replace({ 1 :'Pclass_A' , 2:'Pclass_B' , 3:'Pclass_C'} , inplace =True)

dummies_pclass = pd.get_dummies(df_train['Pclass'])

df_train = pd.concat([df_train, dummies_pclass], axis=1)





df_test.Pclass.replace({ 1 :'Pclass_A' , 2:'Pclass_B' , 3:'Pclass_C'} , inplace =True)

dummies_pclass = pd.get_dummies(df_test['Pclass'])

df_test = pd.concat([df_test, dummies_pclass], axis=1)
df_train["Name"].head()
# Create a family size descriptor from SibSp and Parch

df_train["Fsize"] = df_train["SibSp"] + df_train["Parch"] + 1

df_test["Fsize"] = df_test["SibSp"] + df_test["Parch"] + 1
# Create new feature of family size

df_train['Single'] = df_train['Fsize'].map(lambda s: 1 if s == 1 else 0)

df_train['SmallF'] = df_train['Fsize'].map(lambda s: 1 if  s == 2  else 0)

df_train['MedF'] = df_train['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

df_train['LargeF'] = df_train['Fsize'].map(lambda s: 1 if s >= 5 else 0)



df_test['Single'] = df_test['Fsize'].map(lambda s: 1 if s == 1 else 0)

df_test['SmallF'] = df_test['Fsize'].map(lambda s: 1 if  s == 2  else 0)

df_test['MedF'] = df_test['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

df_test['LargeF'] = df_test['Fsize'].map(lambda s: 1 if s >= 5 else 0)
df_train.sample(10)
df_train = df_train.drop(["Name", "Embarked", "Cabin", "Pclass", "Survived", "Ticket"], axis=1)

df_test = df_test.drop(["Name", "Embarked", "Cabin", "Pclass", "Ticket"], axis=1)
df_train.sample(10)
df_test.sample(10)
df_train = df_train.fillna(df_train.median())

df_test = df_test.fillna(df_test.median())
#scaler = MinMaxScaler()

#train = scaler.fit_transform(df_train)

#test = scaler.transform(df_test)
# Create logistic regression object

model = LogisticRegression()

cross_val_score(model , df_train , y_train , cv=5)
# Create random forest classifier object

model = RandomForestClassifier(bootstrap= True , min_samples_leaf= 3, n_estimators = 500 ,

                               min_samples_split = 10, max_features = "sqrt", max_depth= 6)

cross_val_score(model , df_train , y_train , cv=5)
# Train the model using the training sets

result = model.fit(df_train, y_train)



# Make predictions using the testing set

prediction = model.predict(df_test)
model = SVC(C=4)

cross_val_score(model , df_train , y_train , cv=5)
ans = pd.DataFrame({'PassengerId' : df_test.index , 'Survived': prediction})

ans.to_csv('submit.csv', index = False)

ans.sample(10)