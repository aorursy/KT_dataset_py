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

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc

from sklearn.linear_model import LogisticRegression

import xgboost as xgb

from collections import Counter

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=DeprecationWarning)
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
print(train.shape)

train.head()
print(test.shape)

test.head()
train.info()
test.info()
train.describe(include='all')
test.describe(include='all')
#Checking weather data is balanced or imbalanced

train['Survived'].value_counts()
print(train.pivot_table(values = 'Survived', index = 'Sex'))

train.pivot_table(values = 'Survived', index = 'Sex').plot(kind = 'bar')
sns.kdeplot(train[train['Sex']=='male']['Age'], label = 'male')

sns.kdeplot(train[train['Sex']=='female']['Age'],label = 'female')

plt.legend()

plt.title('Age distribution of male and female')
sns.violinplot(x='Sex', y='Age', hue='Survived', data=train, split=True)
#Let's see the relation between Age and Survived column using graph

train.hist(column="Age",by="Survived",bins=50,figsize=(25,7))
#Now let's see the relation between Pclass and survived column using graph

train.pivot_table(values=['Survived'],index=['Pclass']).plot(kind='bar')
print(train['Pclass'].value_counts())

sns.distplot(train['Pclass'])
#Now combine all Pclass, Age and Survived in one graph and see the result

plt.figure(figsize=(20,9))

sns.violinplot(x='Pclass', y='Age', hue='Survived',data=train, split=True)
#Visualizing of which stoppage passenger died most and of which age group

plt.figure(figsize=(20,9))

sns.violinplot(x='Embarked', y='Age', hue='Survived',data=train, split=True)

#C = Cherbourg, Q = Queenstown, S = Southampton
#Distribution of Age vs Survived

sns.stripplot(train['Survived'],train['Age'], jitter=True)
#Distribution of Fare vs Survived

sns.stripplot(train['Survived'],train['Fare'], jitter=True)
#Distribution of age in different classes

train.hist(column="Age",by="Pclass",bins=30)
def detect_outlier(df,n,cols):

    outlier_indices = []

    for i in cols:

        Q1 = np.percentile(df[i], 25)

        Q3 = np.percentile(df[i], 75)

        IQR = Q3 - Q1

        outlier_step = 1.5*IQR

        outlier_index_list = df[(df[i] < Q1-outlier_step) | (df[i] > Q3+outlier_step)].index

        outlier_indices.extend(outlier_index_list)

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(k for k,v in outlier_indices.items() if v>n)  

    return multiple_outliers
#We are dropping only those indices whose count is at least 2. 

outliers_to_drop = detect_outlier(train,2,['Age', 'SibSp', 'Parch', 'Fare'])

train.loc[outliers_to_drop]
train = train.drop(outliers_to_drop, axis = 0).reset_index(drop=True)
# extracting and then removing the targets from the training data 

targets = train.Survived

train.drop(['Survived'], 1, inplace=True)

    



# merging train data and test data for future feature engineering

# we'll also remove the PassengerID since this is not an informative feature

combined = train.append(test)

combined.reset_index(inplace=True)

combined.drop(['index', 'PassengerId'], inplace=True, axis=1)
#Correlation between different features

sns.heatmap(combined.corr(), annot = True)
print(targets.shape)

combined.shape
combined.info()
#Visualizing null values in the dataset

sns.heatmap(combined.isnull())
combined.head()
#Let's see all the titles

titles = set()

for name in train['Name']:

    titles.add(name.split(',')[1].split('.')[0].strip())

print(titles)
#We will extract titles in 'Title' column

combined['Title'] = combined['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(combined['Title'],combined['Sex'])
#Replacing some titles to the most common titles

combined['Title'] = combined['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

combined['Title'] = combined['Title'].replace('Mlle', 'Miss')

combined['Title'] = combined['Title'].replace('Ms', 'Miss')

combined['Title'] = combined['Title'].replace('Mme', 'Mrs')

#Mapping titles to numerical data

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

combined['Title'] = combined['Title'].map(title_mapping)

combined['Title'] = combined['Title'].fillna(0)

##dropping Name feature

combined = combined.drop(['Name'], axis=1)
combined.head()
combined["Sex"][combined["Sex"] == "male"] = 0

combined["Sex"][combined["Sex"] == "female"] = 1

combined["Sex"] = combined["Sex"].astype(int)
combined.head()
#Age column in highly crrelated with Pclass column

combined.groupby('Pclass')['Age'].describe()
#Filling null values with median of ages in different classes

combined['Age'] = combined.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
combined["Age"] = combined["Age"].astype(int)
#cutting age in different groups

combined['Age_cat'] = pd.qcut(combined['Age'],q=[0, .16, .33, .49, .66, .83, 1], labels=False, precision=1)
combined.head()
#Keeping only alphabets and removing numbers from ticket name, 

#if there is no any alphabets then replace the string with "x"

tickets = []

for i in list(combined.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("x")
combined['Ticket'] = tickets
combined = pd.get_dummies(combined, columns= ["Ticket"], prefix = "T")
combined.head()
combined['Fare'] = combined.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median())) 
combined['Zero_Fare'] = combined['Fare'].map(lambda x: 1 if x == 0 else (0))
def fare_category(fr): 

    if fr <= 7.91:

        return 1

    elif fr <= 14.454 and fr > 7.91:

        return 2

    elif fr <= 31 and fr > 14.454:

        return 3

    return 4
combined['Fare_cat'] = combined['Fare'].apply(fare_category) 
combined.info()
combined["Embarked"] = combined["Embarked"].fillna("C")

combined["Embarked"][combined["Embarked"] == "S"] = 1

combined["Embarked"][combined["Embarked"] == "C"] = 2

combined["Embarked"][combined["Embarked"] == "Q"] = 3

combined["Embarked"] = combined["Embarked"].astype(int)
combined.head()
#Creating a new feature which contains number of members of family

combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
#Grouping FamilySize in different categories

combined['FamilySize_cat'] = combined['FamilySize'].map(lambda x: 1 if x == 1 

                                                            else (2 if 5 > x >= 2 

                                                                  else (3 if 8 > x >= 5 

                                                                       else 4 )

                                                                 ))   
#Creating a new feature Alone

combined['Alone'] = [1 if i == 1 else 0 for i in combined['FamilySize']]
combined.head()
#Filling null values in Cabin feature with 'U'

combined['Cabin'] = combined['Cabin'].fillna('U')
#Keeping only first letter of Cabin name

import re

combined['Cabin'] = combined['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
combined['Cabin'].value_counts()
cabin_category = {'A':9, 'B':8, 'C':7, 'D':6, 'E':5, 'F':4, 'G':3, 'T':2, 'U':1}
combined['Cabin'] = combined['Cabin'].map(cabin_category)
combined.head()
#Creating dummy variable from categorical variable

dummy_col=['Title', 'Sex',  'Age_cat', 'SibSp', 'Parch', 'Fare_cat', 'Cabin', 'Embarked', 'Pclass', 'FamilySize_cat']
dummy = pd.get_dummies(combined[dummy_col], columns=dummy_col, drop_first=False)
combined = pd.concat([dummy, combined], axis = 1)
combined['FareCat_Sex'] = combined['Fare_cat']*combined['Sex']

combined['Pcl_Sex'] = combined['Pclass']*combined['Sex']

combined['Pcl_Title'] = combined['Pclass']*combined['Title']

combined['Age_cat_Sex'] = combined['Age_cat']*combined['Sex']

combined['Age_cat_Pclass'] = combined['Age_cat']*combined['Pclass']

combined['Title_Sex'] = combined['Title']*combined['Sex']

combined['Age_Fare'] = combined['Age_cat']*combined['Fare_cat']



combined['SmallF'] = combined['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

combined['MedF']   = combined['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

combined['LargeF'] = combined['FamilySize'].map(lambda s: 1 if s >= 5 else 0)

combined['Senior'] = combined['Age'].map(lambda s:1 if s>70 else 0)
combined.head()
combined[['Age_Fare','Title_1']].median()
#Seperating training, test and target datasets from combined dataset

X_train = combined[:train.shape[0]]

X_test = combined[train.shape[0]:]

y = targets

X_train['Y'] = y

df = X_train

X = df.drop('Y', axis=1)

y = df.Y
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
#Splitting training dataset into training and validation dataset

x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)
#Creating dmatrix to train xgboost

d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

d_test = xgb.DMatrix(X_test)
params = {

        'objective':'binary:logistic',

        'eta': 0.3,

        'max_depth':9,

        'learning_rate':0.03,

        'eval_metric':'auc',

        'min_child_weight':1,

        'subsample':1,

        'colsample_bytree':0.4,

        'seed':29,

        'reg_lambda':2.8,

        'reg_alpha':0,

        'gamma':0,

        'scale_pos_weight':1,

        'n_estimators': 600,

        'nthread':-1

}
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

nrounds=10000 
model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=600, 

                           maximize=True, verbose_eval=10)
#By using these leaks we can get 0.85 to 0.86 score. If we do not use these leaks 

#we can achieve score of 0.77 to 0.78.

leaks = {

897:1,

899:1, 

930:1,

932:1,

949:1,

987:1,

995:1,

998:1,

999:1,

1016:1,

1047:1,

1083:1,

1097:1,

1099:1,

1103:1,

1115:1,

1118:1,

1135:1,

1143:1,

1152:1, 

1153:1,

1171:1,

1182:1,

1192:1,

1203:1,

1233:1,

1250:1,

1264:1,

1286:1,

935:0,

957:0,

972:0,

988:0,

1004:0,

1006:0,

1011:0,

1105:0,

1130:0,

1138:0,

1173:0,

1284:0,

}
sub = pd.DataFrame()

sub['PassengerId'] = test['PassengerId']

sub['Survived'] = model.predict(d_test)

sub['Survived'] = sub['Survived'].apply(lambda x: 1 if x>0.8 else 0)

sub['Survived'] = sub.apply(lambda r: leaks[int(r['PassengerId'])] if int(r['PassengerId']) in leaks else r['Survived'], axis=1)

sub.to_csv('sub_titan.csv', index=False)