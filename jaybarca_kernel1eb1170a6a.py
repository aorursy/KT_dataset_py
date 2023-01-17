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



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

train_data.shape
train_data.isnull().sum()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

new_age=pd.DataFrame(imputer.fit_transform(train_data.Age.values.reshape(-1,1)))

train_data['Age'] = new_age

train_data.isnull().sum()

train_data.set_index('PassengerId',inplace=True)

## get dummy variables for Column sex and embarked since they are categorical value.

train_data = pd.get_dummies(train_data, columns=["Sex"], drop_first=True)

train_data = pd.get_dummies(train_data, columns=["Embarked"],drop_first=True)

#Mapping the data.

train_data['Fare'] = train_data['Fare'].astype(int)

train_data.loc[train_data.Fare<=7.91,'Fare']=0

train_data.loc[(train_data.Fare>7.91) &(train_data.Fare<=14.454),'Fare']=1

train_data.loc[(train_data.Fare>14.454)&(train_data.Fare<=31),'Fare']=2

train_data.loc[(train_data.Fare>31),'Fare']=3



train_data['Age']=train_data['Age'].astype(int)

train_data.loc[ train_data['Age'] <= 16, 'Age']= 0

train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 32), 'Age'] = 1

train_data.loc[(train_data['Age'] > 32) & (train_data['Age'] <= 48), 'Age'] = 2

train_data.loc[(train_data['Age'] > 48) & (train_data['Age'] <= 64), 'Age'] = 3

train_data.loc[train_data['Age'] > 64, 'Age'] = 4
# In our data the Ticket and Cabin,Name are the base less,leds to the false prediction so Drop both of them.

train_data.drop(['Ticket','Cabin','Name'],axis=1,inplace=True)

train_data.head()
print(type(train_data.Age))
train_data
train_data.Survived.value_counts()/len(train_data)*100
train_data.describe()
train_data.groupby('Survived').mean()
train_data.groupby('Sex_male').mean()
train_data.corr()
import matplotlib.pyplot as plt

import seaborn as sns



plt.subplots(figsize = (15,8))

sns.heatmap(train_data.corr(), annot=True,cmap="PiYG")

plt.title("Correlations Among Features", fontsize = 20);
plt.subplots(figsize = (15,8))

sns.barplot(x = "Sex_male", y = "Survived", data=train_data, edgecolor=(0,0,0), linewidth=2)

plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 25)

labels = ['Female', 'Male']

plt.ylabel("% of passenger survived", fontsize = 15)

plt.xlabel("Gender",fontsize = 15)

plt.xticks(sorted(train_data.Sex_male.unique()), labels)
sns.set(style='darkgrid')

plt.subplots(figsize = (15,8))

ax=sns.countplot(x='Sex_male',data=train_data,hue='Survived',edgecolor=(0,0,0),linewidth=2)

train_data.shape

## Fixing title, xlabel and ylabel

plt.title('Passenger distribution of survived vs not-survived',fontsize=25)

plt.xlabel('Gender',fontsize=15)

plt.ylabel("# of Passenger Survived", fontsize = 15)

labels = ['Female', 'Male']

#Fixing xticks.

plt.xticks(sorted(train_data.Survived.unique()),labels)

## Fixing legends

leg = ax.get_legend()

leg.set_title('Survived')

legs=leg.texts

legs[0].set_text('No')

legs[1].set_text('Yes')
train_data.head(4)
plt.subplots(figsize = (10,10))

ax=sns.countplot(x='Pclass',hue='Survived',data=train_data)

plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)

leg=ax.get_legend()

leg.set_title('Survival')

legs=leg.texts



legs[0].set_text('No')

legs[1].set_text("yes")
plt.subplots(figsize=(10,8))

sns.kdeplot(train_data.loc[(train_data['Survived'] == 0),'Pclass'],shade=True,color='r',label='Not Survived')

ax=sns.kdeplot(train_data.loc[(train_data['Survived'] == 1),'Pclass'],shade=True,color='b',label='Survived' )



labels = ['First', 'Second', 'Third']

plt.xticks(sorted(train_data.Pclass.unique()),labels)
plt.subplots(figsize=(15,10))



ax=sns.kdeplot(train_data.loc[(train_data['Survived'] == 0),'Fare'],color='r',shade=True,label='Not Survived')

ax=sns.kdeplot(train_data.loc[(train_data['Survived'] == 1),'Fare'],color='b',shade=True,label='Survived' )

plt.title('Fare Distribution Survived vs Non Survived',fontsize=25)

plt.ylabel('Frequency of Passenger Survived',fontsize=20)

plt.xlabel('Fare',fontsize=20)
#fig,axs=plt.subplots(nrows=2)

fig,axs=plt.subplots(figsize=(10,8))

sns.set_style(style='darkgrid')

sns.kdeplot(train_data.loc[(train_data['Survived']==0),'Age'],color='r',shade=True,label='Not Survived')

sns.kdeplot(train_data.loc[(train_data['Survived']==1),'Age'],color='b',shade=True,label='Survived')
train_X=train_data.drop('Survived',axis=1)

train_y=train_data['Survived'].astype(int)
from sklearn.svm import SVC

classifier=SVC()

classifier.fit(train_X,train_y)
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
test_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

test_new_age=pd.DataFrame(test_imputer.fit_transform(test_data.Age.values.reshape(-1,1)))

test_new_fare=pd.DataFrame(test_imputer.fit_transform(test_data.Fare.values.reshape(-1,1)))

test_data['Age'] = test_new_age

test_data['Fare'] = test_new_fare

test_data.isnull().sum()
test_data.set_index('PassengerId',inplace=True)

## get dummy variables for Column sex and embarked since they are categorical value.

test_data = pd.get_dummies(test_data, columns=["Sex"], drop_first=True)

test_data = pd.get_dummies(test_data, columns=["Embarked"],drop_first=True)

test_data['Fare'] = test_data['Fare'].astype(int)

test_data.loc[test_data.Fare<=7.91,'Fare']=0

test_data.loc[(test_data.Fare>7.91) &(test_data.Fare<=14.454),'Fare']=1

test_data.loc[(test_data.Fare>14.454)&(test_data.Fare<=31),'Fare']=2

test_data.loc[(test_data.Fare>31),'Fare']=3



test_data['Age']=test_data['Age'].astype(int)

test_data.loc[ test_data['Age'] <= 16, 'Age']= 0

test_data.loc[(test_data['Age'] > 16) & (test_data['Age'] <= 32), 'Age'] = 1

test_data.loc[(test_data['Age'] > 32) & (test_data['Age'] <= 48), 'Age'] = 2

test_data.loc[(test_data['Age'] > 48) & (test_data['Age'] <= 64), 'Age'] = 3

test_data.loc[test_data['Age'] > 64, 'Age'] = 4
test_data.drop(['Ticket','Cabin','Name'],axis=1,inplace=True)

test_data.head()
predictions = classifier.predict(test_data)

print(predictions)

print(len(predictions))



print(test_data.PassengerId)
output = pd.DataFrame({'PassengerId': test_data.index, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")