# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import sys

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# Data Visualisation tools

import seaborn as sns

import matplotlib.pyplot as plt
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.shape
df_train.head()
# Here we can see which features have missing data ie no. of non-null data < no. of entries

df_train.info()

df_test.shape
plt.figure(figsize=(10,6))

sns.swarmplot(x=df_train['Survived'], y=df_train['Age'])

plt.title("Assume default 0 = false")
plt.figure(figsize=(10,6))

sns.swarmplot(x=df_train['Survived'], y=df_train['Fare'])

plt.title("Fare VS chance of living")
df_train.groupby(['Sex','Survived'])['Survived'].count()
thisisjustaname,bleb=plt.subplots(1,2,figsize=(18,8))

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=bleb[0])

bleb[0].set_title('No. of Passengers by Pclass')

bleb[0].set_xlabel('PClass')

bleb[0].set_ylabel('No. of Passengers')



sns.countplot('Pclass',hue='Survived',data=df_train,ax=bleb[1])

bleb[1].set_title('Pclass Dead VS Alive')

#plt.show()
sns.pointplot('Pclass','Survived',hue='Sex',data=df_train)

#sns.kdeplot(data=df_train['Age'], shade=True)
sns.FacetGrid(df_train, col='Survived').map(plt.hist, 'Age', bins = 30)

sns.FacetGrid(df_train, col='Survived').map(plt.hist, 'Age', bins = 80)
df_test.head()
df_test.info()
# Drop cabin columns VS replace null by guessing cabin based on fare??

df_train.drop(['Ticket', 'Cabin','Name'], axis=1)

df_test.drop(['Ticket', 'Cabin','Name'], axis=1)
full_data = [df_train, df_test]

# Remove all NULLS in the Age column

#for dataset in full_data:

#    dataset['Age'] = dataset['Age'].fillna(df_train['Age'].median())

# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Remove all NULLS in the Fare column

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(df_train['Fare'].median())
for dataset in full_data:

    age_avg 	   = dataset['Age'].mean()

    age_std 	   = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)
df_train.info()
df_test.info()
import re as re

def get_title(name):

	title_search = re.search(' ([A-Za-z]+)\.', name)

	# If the title exists, extract and return it.

	if title_search:

		return title_search.group(1)

	return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



print (df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
for dataset in full_data:

    age_avg 	   = dataset['Age'].mean()

    age_std 	   = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

df_train['CategoricalAge'] = pd.cut(df_train['Age'], 5)



print (df_train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
# Method 1 Using scikit learn to label-encode categorical variables

import sklearn

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

label = LabelEncoder()

#for dataset in full_data:    

#    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

#    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])



df_train.head()
# Method 2 Using panda to one-hot encode categorical variables



df_train = pd.get_dummies(df_train, columns=["Sex","Embarked","Title"])

df_test = pd.get_dummies(df_test, columns=["Sex","Embarked","Title"])



for dataset in full_data:

   # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
#for dataset in full_data:

#    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

#    dataset['Title'] = dataset['Title'].map(title_mapping)

#    dataset['Title'] = dataset['Title'].fillna(0)



df_train.head()
#Splitting training data into train/validation

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['Survived','Ticket', 'Cabin','Name','CategoricalAge'], axis = 1), df_train['Survived'],test_size = .33, random_state=0)
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

import xgboost as xgb

from xgboost import XGBClassifier



classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

	AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression(),

    XGBClassifier()]



log_cols = ["Classifier", "Accuracy"]

log 	 = pd.DataFrame(columns=log_cols)



acc_dict = {}



for clf in classifiers:

		name = clf.__class__.__name__

		clf.fit(X_train, y_train)

		train_predictions = clf.predict(X_test)

		acc = accuracy_score(y_test, train_predictions)

		if name in acc_dict:

			acc_dict[name] += acc

		else:

			acc_dict[name] = acc

            

for clf in acc_dict:

	acc_dict[clf] = acc_dict[clf] / 10.0

	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

	log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
clf = XGBClassifier()

clf.fit(X_train, y_train)

result = clf.predict(df_test.drop(['Ticket', 'Cabin','Name'], axis = 1))

submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": result

    })

submission.to_csv('submission.csv', index=False)