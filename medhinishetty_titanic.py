# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output

data = pd.read_csv("/kaggle/input/titanic/train.csv")

test=pd.read_csv("/kaggle/input/titanic/test.csv")

data.head()
#Total number of columns and rows

data.shape
name=[]

cnt=[]

for column in data.columns:

    name.append(column)

    cnt.append(data[column].isna().sum())

tmp=pd.DataFrame(cnt,name)

print('Count of Na values in dataset')

tmp
#test

name=[]

cnt=[]

for column in test.columns:

    name.append(column)

    cnt.append(test[column].isna().sum())

tmp=pd.DataFrame(cnt,name)

print('Count of Na values in dataset')

tmp
#from above we get to know that our dataset contain 3 columns with Na values so we ha ve to  handle this Na

data.Age=data.Age.fillna(data.Age.mean())

data.Embarked=data.Embarked.fillna(data.Embarked.mode()[0])

test.Age=test.Age.fillna(test.Age.mean())
(data.Cabin.isnull().sum()/(data.shape[0]))*100
data=data.drop(['Cabin'],1)

data.sample(3)
# Now we will do analyasis 

# find all unique value from columns

l=[]

for column in data.columns:

    if len(data[column].unique())< 42:

        l.append(column)

for column in l:

    print(column,end='\n')

    for i in data[column].unique():

        print('\t',i)

    print()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

sns.countplot('Survived',data=data).set_title('Survival Count')

sns.countplot(data['Embarked'],hue=data['Survived'])

plt.show()
sns.countplot(data['Parch'],hue=data['Survived'])

plt.show()
sns.countplot(data['SibSp'],hue=data['Survived'])

plt.show()
sns.countplot(data['Pclass'],hue=data['Survived'])

plt.show()
sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="bar", data=data);
a1=data[(data.Age <= 10) & (data.Survived == 1)].count()[0]

a2=data[((data.Age>10)&(data.Age<=20))&(data.Survived==1)].count()[0]

a3=data[((data.Age>20)&(data.Age<=30))&(data.Survived==1)].count()[0]

a4=data[((data.Age>30)&(data.Age<=40))&(data.Survived==1)].count()[0]

a5=data[((data.Age>40)&(data.Age<=50))&(data.Survived==1)].count()[0]

a6=data[((data.Age>50)&(data.Age<=60))&(data.Survived==1)].count()[0]

a7=data[((data.Age>60)&(data.Age<=70))&(data.Survived==1)].count()[0]

a8=data[((data.Age>70)&(data.Age<=80))&(data.Survived==1)].count()[0]

a9=data[(data.Age>80)&(data.Survived==1)].count()[0]

plt.figure(figsize=(8,6))

plt.pie([a1,a2,a3,a4,a5,a6,a7,a8,a9],labels=['0-10 ','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80 above'],autopct='%1.1f%%', shadow=True, startangle=140,radius=1.6)

plt.show()

         
data=data.drop(['Name','Ticket','Fare','Parch'],1)

test=test.drop(['Name','Ticket','Fare','Parch','Cabin'],1)
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

data['Sex']= le.fit_transform(data['Sex'])

data['Embarked']= le.fit_transform(data['Embarked'])

test['Sex']= le.fit_transform(test['Sex'])

test['Embarked']= le.fit_transform(test['Embarked'])

x=data.drop(['Survived'],1)

y=data['Survived']




from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=1300, min_samples_leaf=16,bootstrap = True,

                               max_features = 'sqrt')

# Fit on training data

model.fit(x,y)



prediction=model.predict(test)
sub=pd.read_csv('../input/titanic/gender_submission.csv')

Yt=sub['Survived'].values
from sklearn.metrics import precision_score,accuracy_score



print("Accuaracy:",accuracy_score(Yt, prediction))



print("Precision:", precision_score(Yt, prediction))

testf=pd.read_csv('../input/titanic/test.csv')
submission = pd.DataFrame({'PassengerId':testf['PassengerId'],'Survived':prediction})



#Visualize the first 5 rows

submission.head()

filename = 'Titanic1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)