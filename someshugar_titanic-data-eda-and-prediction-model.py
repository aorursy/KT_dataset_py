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



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# import datasets

df_train = pd.read_csv("../input/train.csv",index_col="PassengerId")

df_test = pd.read_csv("../input/test.csv",index_col="PassengerId")
# view first five lines of training data

df_train.head()
df_test.head()
df_train.info()
df_test.info()
df_test['Survived'] =- 999
total_df = pd.concat((df_train, df_test),axis=0,sort=True)
total_df.info()
total_df.iloc[5:11,0:7]
total_df.tail()
male_df = total_df[total_df['Sex'] == 'male']

female_df = total_df[total_df['Sex'] == 'female']
male_df.info()
female_df.info()
total_df.describe(include='all')
%matplotlib inline

total_df.Fare.plot(kind='box')
%matplotlib inline

total_df.Age.plot(kind='box')
total_df.Sex.value_counts()
total_df.Survived.value_counts()
total_df[total_df.Survived!= -999].Survived.value_counts()
total_df.Pclass.value_counts()
total_df.Pclass.value_counts().plot(kind='bar')
total_df.Age.plot(kind='hist',bins=20,color='g')
total_df.Age.plot(kind='kde',color='g')
total_df.Fare.plot(kind='hist',color='r',bins=20)
total_df.Fare.plot(kind='kde',color='r')
total_df.Age.skew()
total_df.Fare.skew()
total_df.plot.scatter(x='Age',y='Fare',alpha=0.4)
total_df.plot.scatter(x='Pclass',y='Fare',alpha=0.1)
total_df.groupby(['Pclass'])[['Age','Fare']].median()
total_df.groupby(['Pclass']).agg({"Age" : 'mean' ,'Fare' : 'mean'})
total_df.groupby(['Pclass','Embarked']).agg({"Age" : 'median' ,'Fare' : 'median'})
pd.crosstab(total_df.Sex,total_df.Pclass)
pd.crosstab(total_df.Sex,total_df.Pclass).plot(kind='bar')
total_df.pivot_table(index='Sex',columns='Pclass',values='Age',aggfunc='mean')
# Data Munging
total_df.info()
total_df[total_df.Embarked.isnull()]
total_df.Embarked.value_counts()
pd.crosstab(total_df[total_df.Survived != -999].Embarked,total_df[total_df.Survived != -999].Survived)
total_df.pivot_table(index='Pclass',columns='Embarked',values='Fare',aggfunc='median')
total_df.Embarked.fillna('C',inplace=True)
total_df.info()
total_df[total_df.Fare.isnull()]
total_df[(total_df.Pclass ==3) & (total_df.Embarked=='S')].Fare.median()
total_df[total_df.Age.isnull()].count()
print(total_df.Age.mean())

print(total_df.Age.median())

print(total_df.Age.mode())
total_df[total_df.Sex == 'male'].Age.median()
total_df[total_df.Sex == 'female'].Age.median()
total_df[total_df.Sex == 'male'].Age.median()
total_df[total_df.Age.notnull()].boxplot('Age','Pclass')
Median_Age = total_df.groupby('Pclass').Age.transform('median')
total_df.info()
name = total_df.loc[1,'Name'].split(',')

name[1].split('.')[0]
def getsalutation(name):

        sal = name.split(',')[1].split('.')[0].strip()

        return sal
total_df.Name.map(lambda x:getsalutation(x)).unique()
def getTitles(name):

    dicti={'mr':'Mr', 'mrs':'Mrs', 'miss':'Miss', 'master':'Master', 'don':'Sir', 'rev':'Sir', 'dr':'officer', 'mme':'Mrs', 

          'ms':'Mrs',

       'major':'officer', 'lady':'Lady', 'sir':'Sir', 'mlle':'Miss', 'col':'officer', 'capt':'officer', 'the countess':'Lady',

       'jonkheer':'Sir', 'dona':'Lady'}

    sal = (name.split(',')[1].split('.')[0].strip()).lower()

    return dicti[sal]
total_df['Title']=total_df.Name.map(lambda x:getTitles(x))
total_df[['Title','Name']].head()
total_df[total_df.Age.notnull()].boxplot('Age','Title')
Median_Age_title = total_df.groupby('Title').Age.transform('median')

Median_Age_title.unique()
total_df.Age.fillna(Median_Age_title, inplace=True)
total_df.info()
total_df['Age'].plot(kind='hist',bins=20)
total_df.Fare.plot(kind='box')
total_df[total_df.Fare == total_df.Fare.max()]
logfare = np.log(total_df.Fare)
pd.qcut(total_df.Fare,5)
total_df.isnull().sum()
total_df['Fare'].fillna(total_df['Fare'].mean(), inplace = True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
total_df.shape
total_df['Embarked'] = le.fit_transform(total_df['Embarked'])
total_df['FamilySize'] = total_df['SibSp'] + total_df['Parch']
def correlation_heatmap(df, method):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(method=method),

        cmap = colormap,

        square=True, 

        annot=True, 

        annot_kws={'fontsize':9 }

    )

    

    plt.title('Correlation Matrix', y=1.05, size=15)
correlation_heatmap(total_df, 'pearson')
# Drop low corrlations and high cardinality

to_drop = ['Ticket', 'Name', 'Title', 'Age','SibSp', 'Parch', 'FamilySize', 'Embarked']

#to_drop = ['Ticket', 'Name']

total_df = total_df.drop(to_drop, axis=1, inplace=False)
total_df.info()
total_df["CabinBool"] = (total_df["Cabin"].notnull().astype('int'))
total_df['Deck'] = total_df.Cabin.str.extract('([a-zA-Z]+)', expand=False)

total_df[['Cabin', 'Deck']].sample(10)

total_df['Deck'] = total_df['Deck'].fillna('Z')

total_df = total_df.drop(['Cabin'], axis=1)



# label

total_df['Deck'] = le.fit_transform(total_df['Deck'])
total_df.info()
total_df.head()
# Creade dummy variables for Sex and drop original, as well as an unnecessary column (male or female)

total_df = total_df.join(pd.get_dummies(total_df['Sex']))

total_df.drop(['Sex', 'male'], inplace=True, axis=1)
train = total_df[total_df['Survived'] != -999]

test = total_df[total_df['Survived'] == -999]
X_train, y_train = train.loc[:, train.columns != 'Survived'], train.loc[:, train.columns == 'Survived']

X_test = test.loc[:, train.columns != 'Survived']
X_test.index
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



# Split training data into training and validation set

train_X, val_X, train_y, val_y = train_test_split(X_train, y_train, random_state=1)



# Initialize model

model = RandomForestClassifier(n_estimators=100)

# Fit data

model.fit(train_X, train_y)

# Calc accuracy

acc = accuracy_score(model.predict(val_X), val_y)

print("Validation accuracy for Random Forest Model: {:.6f}".format(acc))

y_pred = pd.DataFrame(model.predict(X_test),index=X_test.index.copy(),columns=['Survived'])

pred= y_pred.to_csv('survived.csv', index=True)

importances = model.feature_importances_

sns.barplot(importances,train_X.columns)