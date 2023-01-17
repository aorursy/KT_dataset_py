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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
# fetching train.csv and test.csv

train_df=pd.read_csv('/kaggle/input/titanic/train.csv')

test_df=pd.read_csv('/kaggle/input/titanic/test.csv')

train_df.head()
# lets print these

print(train_df.columns)

print(test_df.columns)
# lets take deep observation about data

print(train_df.info())

print(test_df.info())
#lets check for null values in both datset

print(train_df.isnull().sum())

print('\t\t\t')

print(test_df.isnull().sum())
# lets describe more about data

train_df.describe(include='all')
#lets see relation bw different columns

corr_matrix=train_df.corr()

plt.figure(figsize=(15,6))

sns.heatmap(corr_matrix,annot=True)
fig,ax=plt.subplots(2,2,figsize=(15,10))

sns.countplot(train_df['Sex'],ax=ax[0][0])

sns.countplot(train_df['Embarked'],ax=ax[0][1])

sns.countplot(train_df['Pclass'],ax=ax[1][0])

sns.countplot(train_df['SibSp'],ax=ax[1][1])





ax[0][0].set_title('Total no of male and female')

ax[0][1].set_title('Embarked distribution')

ax[1][0].set_title('Passenger class distribution')

ax[1][1].set_title('Sibling or spouse Distribution')



fig,ax=plt.subplots(1,2,figsize=(15,5))

sns.distplot(train_df['Age'],hist=True,ax=ax[0])

sns.distplot(train_df['Fare'],hist=True,ax=ax[1])





ax[0].set_title('Age distribution')

ax[1].set_title('Fare distribution')





plt.figure(figsize=(15,15))

sns.pairplot(train_df)
# check for null value

train_df.isnull().sum()
#we will replace the age columns by finding the mean of age with respect to Gender as well as passenger class 

train_gp=train_df.groupby(['Sex','Pclass'])['Age'].mean()

print(train_gp)

test_gp=test_df.groupby(['Sex','Pclass'])['Age'].mean()

print(test_gp)





# this function will fill null value with the desire mean value

def fillAgeNa(df):

    for i in range(len(df)) : 

        if pd.isnull(df.loc[i, "Age"]):

            if (df.loc[i,'Sex']=='female') and (df.loc[i,'Pclass']==1) :

                df.loc[i,'Age']=37

            elif(df.loc[i,'Sex']=='female') and (df.loc[i,'Pclass']==2) :

                 df.loc[i,'Age']=26

            elif(df.loc[i,'Sex']=='female') and (df.loc[i,'Pclass']==3):

                 df.loc[i,'Age']=22

            elif(df.loc[i,'Sex']=='male') and (df.loc[i,'Pclass']==1):

                 df.loc[i,'Age']=40

            elif(df.loc[i,'Sex']=='male') and (df.loc[i,'Pclass']==2):

                 df.loc[i,'Age']=30

            elif(df.loc[i,'Sex']=='male') and (df.loc[i,'Pclass']==3):

                 df.loc[i,'Age']=25

    return df

            

                

    
ndf=train_df.copy()

train_df=fillAgeNa(ndf)

train_df.isnull().sum()

#similarly for test data we will fill like

ndf=test_df.copy()

test_df=fillAgeNa(ndf)

train_df['Embarked'].fillna('S',inplace=True)
# now check for null values

train_df.isnull().sum()
# filling null value in fare column with mean

test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)

test_df.isnull().sum()

test_df.shape
# dropping cabin columns

train_data=train_df.drop('Cabin',axis=1)

test_data=test_df.drop('Cabin',axis=1)

test_data.shape


        
train_data.drop(['PassengerId','Name','Ticket'],inplace=True,axis=1)

test_data.drop(['PassengerId','Name','Ticket'],inplace=True,axis=1)
#perform encoding for categorical variable

encd1=pd.get_dummies(train_data[['Sex','Embarked']],drop_first=True)

encd2=pd.get_dummies(test_data[['Sex','Embarked']],drop_first=True)
# now lets concat the encoded categorical data

train_data=pd.concat([train_data,encd1],axis=1)

test_data=pd.concat([test_data,encd2],axis=1)

print(train_data.head(2))

print(test_data.head(2))
# now we will count the size of family

train_data['FamilySize']=train_data['SibSp']+train_data['Parch']+1

test_data['FamilySize']=test_data['SibSp']+test_data['Parch']+1

# here we are apply lambda function 

train_data['Isalone']=train_data['FamilySize'].apply(lambda x : 1 if x>1 else 0)

test_data['Isalone']=test_data['FamilySize'].apply(lambda x : 1 if x>1 else 0)
from sklearn.preprocessing import LabelEncoder
train_data
# now lets drop unnecessry columns like Sex and Embarked

train_data.drop(['Sex','Embarked','FamilySize'],axis=1,inplace=True)

test_data.drop(['Sex','Embarked','FamilySize'],axis=1,inplace=True)

scaler=MinMaxScaler()

scaler.fit(train_data[['Fare']])

train_data['Fare']=scaler.transform(train_data[['Fare']])

train_data['Age']=scaler.fit_transform(train_data[['Age']])

test_data['Fare']=scaler.fit_transform(test_data[['Fare']])

test_data['Age']=scaler.fit_transform(test_data[['Age']])
# here we are applying lamda function to check if sibling or spouse is present or not

train_data['SibSp']=train_data['SibSp'].apply(lambda x: 1 if x>0 else 0)

test_data['SibSp']=test_data['SibSp'].apply(lambda x: 1 if x>0 else 0)
# lets create one mopre important feature by multiplying age with pclass

train_data['nw']=train_data['Age']*train_data['Pclass']

test_data['nw']=test_data['Age']*test_data['Pclass']
print(train_data.head(2))

print(test_data.head(2))
# now everything looks good lets divide our data in x_train and y_train

X_train=train_data.drop('Survived',axis=1)

y_train=train_data[['Survived']]

print('shape of x train and y train')

print(X_train.shape,y_train.shape)

X_test=test_data

print('shape of x test')

print(X_test.shape)

model1=RandomForestClassifier()

model2=XGBClassifier()

model3=LogisticRegression()

model4=SVC(kernel='poly',gamma=1,C=0.1)

model5=KNeighborsClassifier(n_neighbors=23,leaf_size=23,p=1)

#model.fit(X_train,y_train)

model=[model1,model2,model3,model4,model5]
c=0

for m in model:

    c+=1

    m.fit(X_train,y_train)

    accur=round(m.score(X_train,y_train)*100,2)

    print('Model',c)

    print('accuracy =',accur)
''' param_grid={'bootstrap': [True, False],

 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

 'max_features': ['auto', 'sqrt'],

 'min_samples_leaf': [1, 2, 4],

 'min_samples_split': [2, 5, 10],

 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}'''

#grid = RandomizedSearchCV(RandomForestClassifier(), param_grid, refit = True, verbose = 3) 

#grid.fit(X_train, y_train)
#grid.best_params_
model=RandomForestClassifier(n_estimators= 2000,

  min_samples_split= 5,

  min_samples_leaf= 2,

  max_features= 'sqrt',

  max_depth= None)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

y_pred=pd.Series(y_pred)

y_pred=y_pred.apply(lambda x: 1 if x else 0)

accur=round(model.score(X_train,y_train)*100,2)

accur

# sample submission file look like this

dataframe=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

dataframe.head()
# here we will create submission file according to the given sample

subm=pd.concat([test_df['PassengerId'],y_pred],axis=1)

subm.rename(columns={'PassengerId':'PassengerId',0:'Survived'},inplace=True)
# lets verify our submission file with sample file

subm.head()
subm['Survived'].value_counts()
subm.to_csv('submission.csv',index=False)