# importing necessary libraries

import pandas as pd

import pandas_profiling

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score

from xgboost import XGBRegressor
# reading train data

df_train=pd.read_csv('../input/titanic/train.csv')

df_train.head()
# reading test data

df_test=pd.read_csv('../input/titanic/test.csv')

df_test.head()
print(f"Test data shape {df_test.shape} ")

print(f"Train data shape {df_train.shape}")
# information related to the data

print("Train data")

display(df_train.info())

print("Test data")

display(df_test.info())
# checking statistical data related to info

print("Train Data")

display(df_train.describe())

print("Test Data")

display(df_test.describe())
pandas_profiling.ProfileReport(df_train)
def plot_bar(df,feat_x,feat_y,normalize=True):

    """ Plot with vertical bars of the requested dataframe and features"""

    

    ct = pd.crosstab(df[feat_x], df[feat_y])

    if normalize == True:

        ct = ct.div(ct.sum(axis=1), axis=0)

    return ct.plot(kind='bar', stacked=True)
plot_bar(df_train,'Pclass','Survived')

plt.title('Pclass VS Survived')
plot_bar(df_train,'Sex','Survived')

plt.title('Sex VS Survived')
# checking for missing values

print("Missing values  in the training data")

display(df_train.isnull().sum())

print("Missing values in the test data")

display(df_test.isnull().sum())
# checking for percent of missing values in the data

print("Train data")

display(df_train.isnull().sum()/len(df_train))

print("Test data")

display(df_test.isnull().sum()/len(df_test))
# cabin data is missing in both the datasets at a considerable amount

# so we can drop the column

df_test.drop('Cabin',axis=1,inplace=True)

df_train.drop('Cabin',axis=1,inplace=True)



display(df_test.head())

display(df_train.head())
# we can replace the missing values with the medai for age

df_test['Age']=df_test['Age'].fillna(df_test['Age'].median())

df_train['Age']=df_train['Age'].fillna(df_train['Age'].median())
# for embarked values , check  which is the place most people embarked from

display(df_train.loc[df_train['Embarked'].isnull()])

df_train['Embarked'].value_counts(normalize=True)
# replacing the missing embarked values with 'S'

df_train['Embarked']=df_train['Embarked'].fillna('S')
# for the missing fare value, simply replace it with the median

df_test['Fare']=df_test.fillna(df_test['Fare'].median())
# printing missing values in the datasets

print("Test data")

display(df_test.isnull().sum())

print("Train data")

display(df_train.isnull().sum())
# finding co-relatoin between survivors

plt.figure(figsize=(15,7))

sns.heatmap(df_train.corr(),annot=True,cmap='YlGnBu')
# from the heatmap we cann find that survival is highly corealed to Pclass and Fare
plt.figure(figsize=(15,7))

sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=df_train)
plt.figure(figsize=(15,7))

sns.countplot(x='Survived',hue='Pclass',data=df_train,palette='rainbow')
from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder() 

df_train['Embarked']= label_encoder.fit_transform(df_train['Embarked']) 

df_train['Sex']= label_encoder.fit_transform(df_train['Sex'])

df_test['Embarked']= label_encoder.fit_transform(df_test['Embarked']) 

df_test['Sex']= label_encoder.fit_transform(df_test['Sex'])
df_train.head()
# dropping non-usefull columns

df_test.drop(['Name','Ticket'],axis=1,inplace=True)

df_train.drop(['Name','Ticket'],axis=1,inplace=True)
df_test.head()
from sklearn.preprocessing import StandardScaler



scaler=StandardScaler()

df_train[['Age','Fare']]=scaler.fit_transform(df_train[['Age','Fare']])

df_test[['Age','Fare']]=scaler.fit_transform(df_test[['Age','Fare']])
df_train.head()
from sklearn.linear_model import LogisticRegression



X_train = df_train.drop(['Survived','PassengerId'], axis=1)

y_train = df_train["Survived"]

X_test  = df_test.drop("PassengerId", axis=1)

X_train.shape, y_train.shape, X_test.shape
LR=LogisticRegression(max_iter=1000,random_state=1)

LR.fit(X_train,y_train)



# making predictions

y_pred=LR.predict(X_test)
# finding accuracy

print("Accuracy:",round(LR.score(X_train, y_train)*100,2))
dfs = pd.read_csv("../input/titanic/gender_submission.csv")

file = {"PassengerId":dfs["PassengerId"],"Survived":y_pred}

file = pd.DataFrame(file)



file.to_csv("submission_Lr.csv",index=False)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

pred = clf.predict(X_test)





print("Accuracy",round(clf.score(X_train,y_train)*100,2))





dfs = pd.read_csv("../input/titanic/gender_submission.csv")

file = {"PassengerId":dfs["PassengerId"],"Survived":pred}

file = pd.DataFrame(file)



file.to_csv("submission_dt.csv",index=False)
train_x,val_x,train_y,val_y=train_test_split(X_train,y_train)
from sklearn.ensemble import RandomForestRegressor



#Create a Gaussian Classifier

clf=RandomForestRegressor(n_estimators=100)



clf.fit(train_x,train_y)

pred=clf.predict(val_x)



print("MEA",mean_absolute_error(pred,val_y))

print("RMSE",mean_squared_error(pred,val_y,squared=False))
#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



pred=clf.predict(X_test)



# dfs = pd.read_csv("../input/titanic/gender_submission.csv")

file1 = {"PassengerId":dfs["PassengerId"],"Survived":pred}

file1 = pd.DataFrame(file)



file.to_csv("submission_Rf.csv",index=False)
file.head()
clf.score(X_train, y_train)

random_forest = round(clf.score(X_train, y_train) * 100, 2)

random_forest
file.head()