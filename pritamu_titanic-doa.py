# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing data
train=pd.read_csv("../input/titanic/train.csv")
test=pd.read_csv("../input/titanic/test.csv")
train.head()
#checking the data types of each columns
train.dtypes
#Describing the data to get more understanding on its stats.
train.describe()
#to check both data tyoe and null values at once among other features
train.info()
#imputing with 'most_frequent' strategy to deal with missing object as well as int values
from sklearn.impute import SimpleImputer

impute=SimpleImputer(strategy='most_frequent')
imp_train=pd.DataFrame(impute.fit_transform(train))
imp_train.columns=train.columns
imp_train.head()
imp_train.info()
imp_test=pd.DataFrame(impute.fit_transform(test))
imp_test.columns=test.columns
imp_test.info()

#After imputing we can see that the data types of all columns have been converted to objects, we need to convert some of them back to
#integer
#First we will drop two columns: Name and Ticket as both of these are useless for making predictions and we dont want any noise.
train_df=imp_train.drop(['Name','Ticket','PassengerId'], axis=1)
train_df.head()
test_df=imp_test.drop(['Name','Ticket','PassengerId'], axis=1)
test_df.head()
test_df.info()
#Filter out the columns with object data types
object_cols=[cols for cols in train_df.columns if train_df[cols].dtypes=='object']
object_cols
#onverting those columns back to integer type
train_df['Age']=train_df['Age'].astype(int)
train_df['Survived']=train_df['Survived'].astype(int)
train_df['Pclass']=train_df['Pclass'].astype(int)
train_df['SibSp']=train_df['SibSp'].astype(int)
train_df['Parch']=train_df['Parch'].astype(int)
train_df['Fare']=train_df['Fare'].astype(int)

test_df['Age']=test_df['Age'].astype(int)
test_df['Pclass']=test_df['Pclass'].astype(int)
test_df['SibSp']=test_df['SibSp'].astype(int)
test_df['Parch']=test_df['Parch'].astype(int)
test_df['Fare']=test_df['Fare'].astype(int)

print(train_df.info())
test_df.info()
#Combining SibSp and Parch to form a single column Family and then drop the two columns
train_df['Famliy']=train_df['SibSp'] + train_df['Parch']
test_df['Family']=test_df['SibSp'] + test_df['Parch']

valid_df=train_df['Survived']
train_df=train_df.drop(['SibSp','Parch','Survived','Cabin'], axis=1)
test_df=test_df.drop(['SibSp','Parch','Cabin'], axis=1)

print(train_df.head())
print(valid_df.head())
print(test_df.head())

tr_df=train_df.copy()
te_df=test_df.copy()
#For Dealing with categorical values

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

LabelEnc=LabelEncoder()
Ohe=OneHotEncoder(handle_unknown='ignore', sparse=False)


train_df.head()
object_cols=[cols for cols in train_df.columns if train_df[cols].dtypes=='object']
object_cols
#LabelEncoding the dataset

label_train_df=train_df
label_test_df=test_df
for cols in set(object_cols):
    label_train_df[cols]=LabelEnc.fit_transform(train_df[cols])
    label_test_df[cols]=LabelEnc.transform(test_df[cols])
    
print(train_df.info())
label_train_df.head()

age_tr_df=label_train_df.copy()
age_te_df=label_test_df.copy()
#Importing various classification models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train,X_valid,y_train,y_valid=train_test_split(label_train_df,valid_df, test_size=0.2, random_state=1)

#Logistic Regression

logreg=LogisticRegression()

logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_valid)

print(accuracy_score(y_valid,y_pred))
#RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=60, max_depth=5)

rfc.fit(X_train,y_train)

rfc_y_pred=rfc.predict(X_valid)

print(accuracy_score(y_valid,rfc_y_pred))
#Support Vector Classifier

svc=SVC()

svc.fit(X_train,y_train)

svc_y_pred=svc.predict(X_valid)

print(accuracy_score(y_valid,svc_y_pred))
#Select the model with the highest accuracy i.e. Logistic Regression
pred=logreg.predict(label_test_df)
#Submit
'''
Submission=pd.DataFrame({"PassengerId":test.PassengerId,"Survived":pred})
Submission.to_csv("Submission.csv", index=False) '''
#Now try out the same models with OneHotEncoding

o_tr_df=pd.DataFrame(Ohe.fit_transform(tr_df[object_cols]))
o_te_df=pd.DataFrame(Ohe.transform(te_df[object_cols]))

o_tr_df.index=tr_df.index
o_te_df.index=te_df.index

num_tr_df=tr_df.drop(object_cols, axis=1)
num_te_df=te_df.drop(object_cols,axis=1)

final_tr=pd.concat([o_tr_df,num_tr_df], axis=1)
final_te=pd.concat([o_te_df,num_te_df], axis=1)


final_tr.head()
#Logistic
X_train,X_valid,y_train,y_valid=train_test_split(final_tr,valid_df, test_size=0.2, random_state=1)

logreg.fit(X_train,y_train)
log_pred=logreg.predict(X_valid)
print(accuracy_score(y_valid,log_pred))
#RandomFOrest
rfc=RandomForestClassifier(n_estimators=100, max_depth=9)
rfc.fit(X_train,y_train)

rfc_pred=rfc.predict(X_valid)

print(accuracy_score(y_valid,rfc_pred))
#SupportVectorMAchines
svc=SVC()

svc.fit(X_train,y_train)

svc_pred=svc.predict(X_valid)

print(accuracy_score(y_valid,svc_pred))
#Highest accuracy: RandomFOrest 

final_pred=rfc.predict(final_te)
Submission=pd.DataFrame({"PassengerId":test.PassengerId,"Survived":final_pred})
Submission.to_csv("Submission2.csv", index=False)
age_tr_df.insert(6,'Survived',train['Survived'])
age_tr_df['AgeBand']=pd.cut(age_tr_df['Age'],5)
age_tr_df[['AgeBand','Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand',ascending=True)

for dataset in [age_tr_df,age_te_df]:
    dataset.loc[dataset['Age']<=16,'Age']=0
    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=32),'Age']=1
    dataset.loc[(dataset['Age']>32) & (dataset['Age']<=48),'Age']=2
    dataset.loc[(dataset['Age']>48) & (dataset['Age']<=64),'Age']=3
    dataset.loc[(dataset['Age']>64) & (dataset['Age']<=80),'Age']=4
    
age_tr_df.head()
#age_tr_df=age_tr_df.drop(['AgeBand'], axis=1)
age_tr_df.insert(5,'Family', age_tr_df['Famliy'])
age_tr_df.drop(['Famliy'], axis=1,inplace=True)
age_tr_df.head()
for dataset in [age_tr_df,age_te_df]:
    dataset['FamilySize']=dataset['Family']+1
age_tr_df[['FamilySize','Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='FamilySize', ascending=True)
age_tr_df.insert(1,'Name',train['Name'])
age_te_df.insert(1,'Name',test['Name'])
print(age_tr_df.head())
for dataset in [age_tr_df,age_te_df]:
    dataset['Title']=dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(age_tr_df['Title'], age_tr_df['Sex'])
for dataset in [age_tr_df, age_te_df]:
    dataset['Title']=dataset['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Rev','Sir'],'Rare')
    dataset['Title']=dataset['Title'].replace(['Mlle','Ms'],'Miss')
    dataset['Title']=dataset['Title'].replace('Mme','Mrs')
    
age_tr_df[['Title','Survived']].groupby(['Title'], as_index=False).mean()
title_map={'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Rare':4}

for dataset in [age_tr_df, age_te_df]:
    dataset['Title']=dataset['Title'].map(title_map)
    dataset['Title']=dataset['Title'].fillna(0)
    
age_tr_df.head()

age_te_df=age_te_df.drop(['Name'], axis=1)
age_tr_df.head()
age_te_df.head()
'''
rand=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
y=age_tr_df['Survived']
X=age_tr_df.drop(['Survived'], axis=1)

X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.2, random_state=1)

rand.fit(X_train,y_train)
predictions=rand.predict(X_valid)

print(accuracy_score(y_valid,predictions))

predictions=rand.predict(age_te_df)


output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission4.csv', index=False)
print("Your submission was successfully saved!")    '''
from xgboost import XGBClassifier

xg_model=XGBClassifier(random_state=0, n_estimators=100,learning_rate=0.9)

xg_model.fit(X_train,y_train, early_stopping_rounds=10, eval_set=[(X_valid,y_valid)])

pred=xg_model.predict(X_valid)

print(accuracy_score(y_valid,pred))


'''
predictions=xg_model.predict(age_te_df)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('xg_submission1.csv', index=False)
print("Your submission was successfully saved!") 
'''
