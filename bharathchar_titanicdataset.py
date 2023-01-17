# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv',index_col=None)

train_data.head()
train_data['PassengerId'].count()
train_data.info()
train_data.describe()
train_data['Age'].isnull().count()
import matplotlib.pyplot as plt
with sns.plotting_context("notebook",font_scale=1):

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(25,15))

    sns.distplot(train_data["Age"].dropna(),bins=80,kde=False,color="red", ax=ax)

    plt.ylabel("Count")

    plt.title("Age Distribution")

sns.pairplot(train_data, hue="Sex");
train_data[:10]
train_data.tail(10)
train_data.isna().sum()
(train_data['Age'].isna().sum() / train_data['PassengerId'].count()) * 100
train_data['Survived'].sum()
(train_data['Survived'].sum() / train_data['PassengerId'].count()) * 100
sns.countplot(x='Survived', data=train_data, hue='Sex',palette='bwr')
train_male_survival = train_data[(train_data['Survived'] == 1) & (train_data['Sex'] == 'male')]

train_male_survival['PassengerId'].count()
train_female_survival = train_data[(train_data['Survived'] == 1) & (train_data['Sex'] == 'female')]

train_female_survival['PassengerId'].count()
train_male_survival['PassengerId'].count() / (train_female_survival['PassengerId'].count()+train_male_survival['PassengerId'].count()) * 100
train_female_survival['PassengerId'].count() / (train_female_survival['PassengerId'].count()+train_male_survival['PassengerId'].count()) * 100
train_data.groupby('Embarked').count()
temp_train = train_data[train_data['Embarked'].notnull()]

temp_train.count()
sns.countplot(x='Survived', data=temp_train, hue='Embarked', palette='bwr')
# temp_train = temp_train[temp_train['Survived']==1].groupby(['Embarked','Sex'],as_index=False).agg({'Cabin':'count'})

temp_train = train_data[train_data.Survived == 1]

temp_train.head()





sns.countplot(x='Pclass', data=temp_train, hue='Sex',palette='bwr')
temp_train = train_data[train_data.Age.notna()]

temp_train['Child_Adult'] = np.where(temp_train.eval("Age <= 16"), "child", "adult")

temp_train.head()
temp_pclass1 = temp_train[temp_train.Pclass == 1]

temp_pclass2 = temp_train[temp_train.Pclass == 2]

temp_pclass3 = temp_train[temp_train.Pclass == 3]

temp_pclass1.head()
print('Pclass 1 survival rate: ' + str((temp_pclass1['Survived'][temp_pclass1.Survived == 1].count()/temp_pclass1['PassengerId'].count())*100))

print('Pclass 2 survival rate: ' + str((temp_pclass2['Survived'][temp_pclass2.Survived == 1].count()/temp_pclass2['PassengerId'].count())*100))

print('Pclass 3 survival rate: ' + str((temp_pclass3['Survived'][temp_pclass3.Survived == 1].count()/temp_pclass3['PassengerId'].count())*100))





print('Total Pclass 1 passengers count: '+ str(temp_pclass1.PassengerId.count()))

print('Total Pclass 2 passengers count: '+ str(temp_pclass2.PassengerId.count()))

print('Total Pclass 3 passengers count: '+ str(temp_pclass3.PassengerId.count()))
sns.countplot(x='Pclass', data=temp_pclass1, hue='Sex',palette='bwr')
sns.countplot(x='Pclass', data=temp_pclass2, hue='Sex',palette='bwr')
sns.countplot(x='Pclass', data=temp_pclass3, hue='Sex',palette='bwr')
temp_pclass1 = temp_pclass1[temp_pclass1.Age.notna()]

temp_pclass2 = temp_pclass2[temp_pclass2.Age.notna()]

temp_pclass3 = temp_pclass3[temp_pclass3.Age.notna()]
sns.countplot(x='Child_Adult', data=temp_pclass1, hue='Survived',palette='BrBG')
sns.countplot(x='Child_Adult', data=temp_pclass2, hue='Survived',palette='BrBG')
sns.countplot(x='Child_Adult', data=temp_pclass3, hue='Survived',palette='BrBG')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
train_data['Age'] = train_data[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train_data.drop('Cabin',axis=1,inplace=True)
train_data.dropna(inplace=True)
sex = pd.get_dummies(train_data['Sex'],drop_first=True)

embark = pd.get_dummies(train_data['Embarked'],drop_first=True)
train_data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train_data = pd.concat([train_data,sex,embark],axis=1)
train_data = pd.concat([train_data,sex,embark],axis=1)
train_data.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data.drop('Survived',axis=1), 

                                                    train_data['Survived'], test_size=0.30, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))