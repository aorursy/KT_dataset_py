import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df1 = pd.read_csv('../input/titanic/test.csv')

df1.head()
df2 = pd.read_csv('../input/titanic/train.csv')

df2.head()
df1.shape
df2.shape
df1.columns
df2.columns
df2['Survived'].value_counts()
df2['Sex'].value_counts()
df2['Parch'].value_counts()
df2['SibSp'].value_counts()
df2['Pclass'].value_counts()
df1.nunique() 
df2.nunique() 
df2.apply(len) 
df2.apply(len, axis = 'columns') 
df2['Fare'].nlargest(10) 
df2['Fare'].nsmallest(10) 
df2['Age'].nlargest(10) 
df2['Age'].nsmallest(10) 
df2['SibSp'].nlargest(10) 
df2['SibSp'].nsmallest(10)
df2['Parch'].nlargest(10) 
df2['Parch'].nsmallest(10) 
df2.describe()
df2.isnull().sum()
plt.style.available
plt.style.use('fivethirtyeight')
sns.heatmap(df2.corr(), annot=True);
df2['SibSp'].value_counts().plot(kind = 'bar', title = 'Count of Siblings or Spouses', color = ['#ff2e63','#fe9881']);

plt.xlabel('Number of Sib/Sp');

plt.ylabel('Count');
plt.bar(df2.index, df2['SibSp'], align = 'center', color = 'green', width=5);

plt.title('Count of the Siblings or Spouses')

plt.xlabel('Index')

plt.ylabel('Number of Siblings or Spouses');
df2['Parch'].value_counts().plot(kind='bar', title = 'Count of Parent or Chilredn', color = '#B33B24');

plt.xlabel('Number of Parch')

plt.ylabel('Count');
df2['Survived'].value_counts().plot(kind = 'bar', title = 'Count for the Survival', color = ['#FE4C40', '#FFCC33'] );

plt.xlabel('Survived or not')

plt.ylabel('Count');
sns.countplot(x = 'Sex', data = df2, palette = 'Blues')

plt.xlabel('Male of Female')

plt.ylabel('Count')

plt.title('Count of Male and Female');
df2['Embarked'].value_counts().plot(kind = 'bar', title = 'Count for the Port of Embarkation', color = ['#B33B24', '#CC553D','#E6735C']);

plt.xlabel('Type of Embarkation')

plt.ylabel('Count');
df2['Pclass'].value_counts().plot(kind = 'bar', title = 'Count for the Pclass', color = ['#FFCC33','#FF6037','#FE4C40']);

plt.xlabel('Type of Pclass')

plt.ylabel('Count');
df2['Fare'].nlargest(10).plot(kind='bar', title = '10 largest Fare', color = ['#C62D42', '#FE6F5E']);

plt.xlabel('Index')

plt.ylabel('Fare');
df2['Age'].nlargest(10).plot(kind='bar', color = ['#5946B2','#9C51B6']);

plt.title('10 largest Ages')

plt.xlabel('Index')

plt.ylabel('Ages');
df2['Age'].nsmallest(10).plot(kind='bar', color = ['#A83731','#AF6E4D'])

plt.title('10 smallest Ages')

plt.xlabel('Index')

plt.ylabel('Ages');
df2['SibSp'].nlargest(10).plot(kind='bar', color = ['#33CC99','#00755E'])

plt.title('Index having largest number of SibSp')

plt.xlabel('Index')

plt.ylabel('Number of SibSp');
df2['Parch'].nlargest(10).plot(kind='bar', color = ['#319177','#0A7E8C'])

plt.title('Index having largest no. of Parch')

plt.xlabel('Index')

plt.ylabel('Number of Parch');
bins = [10,20,30,40,50,60,70,80,90,100]

plt.hist(df2['Age'], bins = bins, edgecolor = 'black', color = '#008080');

plt.title('Count for the range of ages')

plt.xlabel('Ages')

plt.ylabel('Number of Counts');
bins = [10,20,30,40,50,60,70,80,90]

plt.hist(df2['Fare'], bins = bins, edgecolor = 'black', color = '#CD607E');

plt.title('Count for the range of Fare')

plt.xlabel('Fare')

plt.ylabel('Number of Counts');
sns.countplot(x = 'Survived', data = df2, hue = 'Pclass', palette = 'Greens');

plt.title('Survival of people according to Pclass');
sns.countplot(x = 'Survived', data = df2, hue = 'Sex', palette = 'Greys')

plt.title('Survival of people according to sex');
sns.countplot(x = 'Survived', data = df2, hue = 'Embarked', palette = 'Accent');

plt.title('Survival of people according to Embarked');
sns.boxplot(data = df2['Fare'],orient = 'h', palette = 'Blues');
sns.boxplot(data = df2['Age'], orient = 'h', palette = 'Greens');
plt.scatter(df2['Pclass'], df2['Fare'], color = '#676767')

plt.title('Fare according to Pclass')

plt.xlabel('Pclass')

plt.ylabel('Fare');
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.countplot(x='Pclass', data = df2, palette = 'rainbow');

plt.title('Count of Pclass');





plt.subplot(1,2,2)

sns.countplot(x='Survived', data = df2, hue = 'Pclass', palette = 'rainbow');

plt.title('Survival according to Pclass');
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.countplot(x='Sex', data = df2, palette = 'Blues')

plt.title('Count of Sex');



plt.subplot(1,2,2)

sns.countplot(x='Survived', data = df2, hue = 'Sex', palette = 'Blues')

plt.title('Survival according to Sex');
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.countplot(x = 'Embarked', data = df2, palette = 'Greens')

plt.title('Count of Embarkation');



plt.subplot(1,2,2)

sns.countplot(x = 'Survived', data = df2, hue='Embarked', palette = 'Greens')

plt.title('Survival according to point of embarkation');