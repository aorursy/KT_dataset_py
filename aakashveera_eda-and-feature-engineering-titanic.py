import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("/kaggle/input/titanic/train.csv")
df.info()
df.head()
df.drop("PassengerId",axis=1,inplace=True)
sns.countplot(df['Pclass'])
sns.countplot(df['Survived'])
sns.countplot(df['Survived'],hue=df['Sex'],palette='twilight_shifted_r')
sns.countplot(df['Survived'],hue=df['Pclass'],palette='viridis')
sns.countplot(df['SibSp'])
sns.countplot(df['Parch'])
sns.distplot(df['Age'],kde=False,bins=40)
sns.boxplot(df['Age'])
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')
def age_fill(cols):

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
df['Age'] = df[['Age','Pclass']].apply(age_fill,axis=1)
df['Cabin'] = df['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
df['Embarked'].value_counts()
df['Embarked'].fillna(value='S',inplace=True)
df.head()
def extract_title(arg):

    return arg.split(' ')[1]



df['Title'] = df['Name'].apply(extract_title)

#df['Title'] = df['Name'].apply(lambda x: x.split(' ')[1])  equivalent lambda function

df.drop('Name',axis=1,inplace=True)
df.drop(['Ticket','Fare'],axis=1,inplace=True)
#you may also use function approach to convert the data. But let's see with LabelEncoder

from sklearn.preprocessing import LabelEncoder

encoder_sex = LabelEncoder()

encoder_embarked = LabelEncoder()

encoder_title = LabelEncoder()
df['Sex'] = encoder_sex.fit_transform(df['Sex'])

df['Embarked'] = encoder_embarked.fit_transform(df['Embarked'])

df['Title'] = encoder_title.fit_transform(df['Title'])
df['Class_sex'] = df['Pclass'].astype(str) + df['Sex'].astype(str)

encoder_Class_sex = LabelEncoder()

df['Class_sex'] = encoder_Class_sex.fit_transform(df['Class_sex'])
df
test = pd.read_csv("/kaggle/input/titanic/test.csv")
#Don't drop the PassengerId is it neccassary for submission

test.drop(['Ticket','Fare'],axis=1,inplace=True)
test['Age'] = test[['Age','Pclass']].apply(age_fill,axis=1)
test['Title'] = test['Name'].apply(extract_title)

test.drop('Name',axis=1,inplace=True)
test['Cabin'] = test['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
test.head()
test['Sex'] = encoder_sex.transform(test['Sex'])

test['Embarked'] = encoder_embarked.transform(test['Embarked'])
test['Title'].value_counts()
test.loc[test['Title']=='Khalil,','Title'] = 'Mr.'

test.loc[test['Title']=='Palmquist,','Title'] = 'Mr.'

test.loc[test['Title']=='Brito,','Title'] = 'Mr.'
test['Title'] = encoder_title.transform(test['Title'])
test['Class_sex'] = test['Pclass'].astype(str) + test['Sex'].astype(str)

test['Class_sex'] = encoder_Class_sex.transform(test['Class_sex'])
test.to_csv("test.csv",index=False)

df.to_csv("train.csv",index=False)