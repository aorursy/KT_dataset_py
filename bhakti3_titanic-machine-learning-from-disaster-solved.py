import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.head()
test_data.head()
train_data.info()
train_data['Survived'].value_counts()
import seaborn as sb

import matplotlib.pyplot as plt



sb.pairplot(train_data, hue="Survived")
s_class=pd.value_counts(train_data['Survived'])

print(s_class)

s_class.plot(kind='bar',color=["r","b"])
fig, ax = plt.subplots(figsize=(20,10))

corr = train_data.corr() 

sb.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,ax=ax,cmap="YlGnBu",annot=True)
print("Total Missing Valuesi in Age column:",train_data['Age'].isnull().sum())

print("Total Missing Valuesi in Cabin column:",train_data['Cabin'].isnull().sum())

print("Total Missing Valuesi in Embarked column:",train_data['Embarked'].isnull().sum())
train_data=train_data.fillna(0)
print("Total Missing Values in Age column:",train_data['Age'].isnull().sum())

print("Total Missing Values in Cabin column:",train_data['Cabin'].isnull().sum())

print("Total Missing Values in Embarked column:",train_data['Embarked'].isnull().sum())
print("Duplicates Found:",train_data.duplicated().sum())
train_data.describe()
new_train_data=train_data.drop(['Survived','Name','Ticket','Fare'],axis=1)

train_lab=train_data['Survived']
new_train_data['Sex']=new_train_data['Sex'].astype('category')

new_train_data['Sex']=new_train_data['Sex'].cat.codes

new_train_data['Sex'].value_counts()
# new_train_data['Age'].value_counts()
new_train_data['Cabin']=new_train_data['Cabin'].astype('category')

new_train_data['Cabin']=new_train_data['Cabin'].cat.codes

new_train_data['Cabin'].value_counts()
new_train_data['Embarked']=new_train_data['Embarked'].astype('category')

new_train_data['Embarked']=new_train_data['Embarked'].cat.codes

new_train_data['Embarked'].value_counts()
new_train_data.info()

new_train_data.head()