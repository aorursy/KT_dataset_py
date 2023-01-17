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
## Import the useful libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

train_DF = pd.read_csv("/kaggle/input/titanic/train.csv")

test_DF = pd.read_csv("/kaggle/input/titanic/test.csv")
print(sns.distplot(train_DF['Age'].dropna(),kde=False,bins=30))
print(sns.distplot(test_DF['Age'].dropna(),kde=False,bins=30))
plt.figure(figsize=(10,7))

print(sns.boxplot(x='Survived',y='Age',data=train_DF))
plt.figure(figsize=(10,7))

print(sns.boxplot(x='Pclass',y='Age',data=train_DF))
print(sns.heatmap(train_DF.isnull(),yticklabels=False,cmap='viridis'))
print(sns.heatmap(test_DF.isnull(),yticklabels=False,cmap='viridis'))
print("Average_Age_Of_First_Class_Pass",round(train_DF['Age'][train_DF['Pclass']==1].mean()))

print("Average_Age_Of_Second_Class_Pass",round(train_DF['Age'][train_DF['Pclass']==2].mean()))

print("Average_Age_Of_Third_Class_Pass",round(train_DF['Age'][train_DF['Pclass']==3].mean()))

print("\n")

print("Average_Age_Of_First_Class_Pass",round(test_DF['Age'][test_DF['Pclass']==1].mean()))

print("Average_Age_Of_Second_Class_Pass",round(test_DF['Age'][test_DF['Pclass']==2].mean()))

print("Average_Age_Of_Third_Class_Pass",round(test_DF['Age'][test_DF['Pclass']==3].mean()))
## For the train DataSet :



Pclass1_age = round(train_DF['Age'][train_DF['Pclass']==1].mean())

Pclass2_age = round(train_DF['Age'][train_DF['Pclass']==2].mean())

Pclass3_age = round(train_DF['Age'][train_DF['Pclass']==3].mean())



def fix_age_train_DF(col):

    Age=col[0]

    Pclass=col[1]

    

    if pd.isnull(Age):

        

        if Pclass == 1:

            return Pclass1_age

        elif Pclass == 2:

            return Pclass2_age

        else:

            return Pclass3_age

    else:

        return Age
## For the test DataSet :



Pclass1_age = round(test_DF['Age'][test_DF['Pclass']==1].mean())

Pclass2_age = round(test_DF['Age'][test_DF['Pclass']==2].mean())

Pclass3_age = round(test_DF['Age'][test_DF['Pclass']==3].mean())



def fix_age_test_DF(col):

    Age=col[0]

    Pclass=col[1]

    

    if pd.isnull(Age):

        

        if Pclass == 1:

            return Pclass1_age

        elif Pclass == 2:

            return Pclass2_age

        else:

            return Pclass3_age

    else:

        return Age
train_DF['Age'] = train_DF[['Age','Pclass']].apply(fix_age_train_DF,axis=1)

test_DF['Age'] = test_DF[['Age','Pclass']].apply(fix_age_test_DF,axis=1)
print(sns.heatmap(test_DF.isnull(),yticklabels=False,cmap='viridis'))
print(sns.heatmap(test_DF.isnull(),yticklabels=False,cmap='viridis'))
train_DF.drop('Cabin',axis=1,inplace=True)

test_DF.drop('Cabin',axis=1,inplace=True)
train_DF.info()
train_DF.dropna(inplace=True)  ## Drop any more rows with NULL values

test_DF.fillna(value=test_DF['Age'].min(),inplace=True) ## There is one row that is null/NaN ... so replacing with the minimum value.
train_DF.info()

test_DF.info()
sex=pd.get_dummies(train_DF['Sex'],drop_first=True)

sex1=pd.get_dummies(test_DF['Sex'],drop_first=True)



embark=pd.get_dummies(train_DF['Embarked'],drop_first=True)

embark1=pd.get_dummies(test_DF['Embarked'],drop_first=True)



pclass=pd.get_dummies(train_DF['Pclass'],drop_first=True)

pclass1=pd.get_dummies(test_DF['Pclass'],drop_first=True)



train_DF=pd.concat([train_DF,sex,embark,pclass],axis=1)

test_DF=pd.concat([test_DF,sex1,embark1,pclass1],axis=1)
train_DF.head()
test_DF.head()
train_DF.drop(['Sex','Embarked','Name','Ticket','PassengerId','Pclass'],axis=1,inplace=True)

test_DF.drop(['Sex','Embarked','Name','Ticket','Pclass'],axis=1,inplace=True)
print(train_DF.head())

print(test_DF.head())
X_train = train_DF.drop('Survived',axis=1)

y_train = train_DF['Survived']
print(X_train.head())

print(y_train.head())
X_test = test_DF.drop('PassengerId',axis=1)
print(X_test.head())
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(random_state=0)
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
AgeBased_Submission=pd.DataFrame(index=test_DF.PassengerId, data={'Survived': predictions})
print(AgeBased_Submission.head())

print(AgeBased_Submission.info())
AgeBased_Submission.to_csv('/kaggle/working/AgeBased_Submission_2.csv')