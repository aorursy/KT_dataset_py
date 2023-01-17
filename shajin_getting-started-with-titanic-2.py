# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from category_encoders.one_hot import OneHotEncoder

from category_encoders.target_encoder import TargetEncoder

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.feature_selection import RFE

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import StackingClassifier

from scipy.optimize import fmin

from hyperopt import hp, tpe, fmin, Trials

from keras.models import Sequential

from keras.layers import Dense

#import pydotplus as pdot

from IPython.display import Image



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head()
test.head()
submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv", index_col = 0)

submission.head()
print(plt.style.available)
sns.set(font_scale = 1.2)

plt.style.use(['ggplot', 'bmh'])

plt.figure(figsize=(6, 7))

ax = sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data = train)

ax.set_title("Barplot Showing Survival Rate of Male and Female", y = 1.05)

ax.set_xlabel("Gender", fontsize=16)

ax.set_ylabel("Survival Rate", fontsize=16)



plt.show()
women = train.loc[train.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women*100)

men = train.loc[train.Sex == 'male']['Survived']

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men*100)
sns.set(font_scale = 1.1)

plt.style.use('dark_background')

plt.figure(figsize=(14, 6))

plt.subplot(121)

missing_train = train.isnull().sum()

missing_train.sort_values(ascending=False, inplace=True)

missing_train.plot(kind = 'bar')

plt.xlabel("Features", fontsize = 14)

plt.ylabel("Count", fontsize = 14)

plt.title("Missing Values of Train Dataset", y = 1.05)

plt.subplot(122)

missing_test = test.isnull().sum()

missing_test.sort_values(ascending=False, inplace=True)

missing_test.plot(kind = 'bar')

plt.xlabel("Features", fontsize = 14)

plt.ylabel("Count", fontsize = 14)

plt.title("Missing Values of Test Dataset", y = 1.05)

plt.show()

plt.show()
missing_train[missing_train>0]
missing_test[missing_test>0]
train.info()
plt.style.use('seaborn-whitegrid')

count, bin_edges = np.histogram(train.Fare)

train['Fare'].plot(kind = 'hist', xticks = bin_edges, figsize = (8, 5))

plt.xlabel("Fare")

plt.ylabel("Frequency")

plt.title("Distribution of Fare", y = 1.04)

plt.show()
plt.figure(figsize = (8, 5))

sns.distplot(train.Age)

plt.title("Distribution of Age", y = 1.04)

plt.xlabel("Age of Passengers")

plt.ylabel("Frequency")

plt.show()
sns.set(font_scale = 1.2)

plt.style.use('ggplot')

plt.figure(figsize=(4, 5))

ax = sns.barplot(x = 'Embarked', y = 'Survived', data = train)

ax.set_title("Barplot Showing Survival Rate with Respect to Port of Embarkation", y = 1.05)

ax.set_xlabel(ax.get_xlabel(), fontsize=16)

ax.set_ylabel(ax.get_ylabel(), fontsize=16)

plt.show()
plt.figure(figsize = (8, 5))

sns.countplot(train.Embarked)

plt.show()
sns.heatmap(train.corr(), annot = True, cmap = 'BrBG')

plt.show()
full = pd.concat([train, test])

full.head()
from statistics import mode



full['Age'] = full.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

full.Fare = full.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))

full["Embarked"] = full["Embarked"].fillna(mode(full["Embarked"]))

full['Cabin'].fillna('U', inplace=True)

full.isnull().sum()
full.Cabin.unique().tolist()
#Let's engineer the Cabin feature a little bit. We will go through the other steps of feature engineering later

import re



# Extract (first) letter!

full['Cabin'] = full['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

full.Cabin.unique().tolist()
plt.style.use('seaborn-whitegrid')

count, bin_edges = np.histogram(full.Fare)

full['Fare'].plot(kind = 'hist', xticks = bin_edges, figsize = (8, 5))

plt.xlabel("Fare")

plt.ylabel("Frequency")

plt.title("Distribution of Fare", y = 1.04)

plt.show()
plt.figure(figsize = (8, 5))

sns.distplot(full.Age)

plt.title("Distribution of Age", y = 1.04)

plt.xlabel("Age of Passengers")

plt.ylabel("Frequency")

plt.show()
sns.set(font_scale = 1.2)

plt.style.use('ggplot')

plt.figure(figsize=(4, 5))

ax = sns.barplot(x = 'Cabin', y = 'Survived', data = full)

ax.set_title("Barplot Showing Survival Rate with Respect to Cabin Type", y = 1.05)

ax.set_xlabel(ax.get_xlabel(), fontsize=14)

ax.set_ylabel(ax.get_ylabel(), fontsize=14)

plt.show()
sns.countplot(full.Cabin)

plt.show()
full.head()
print("train shape:", train.shape, " test shape:", test.shape)
# Recover test dataset

test = full[full['Survived'].isna()].drop(['Survived'], axis = 1)



# Recover train dataset

train = full[full['Survived'].notna()]
print("train shape:", train.shape, " test shape:", test.shape)
plt.style.use('fivethirtyeight')

plt.figure(figsize=(4, 5))

ax = sns.barplot(x = 'Pclass', y = 'Survived', data = train)

ax.set_title("Barplot Showing Survival Rate with Respect to Passenger Class", y = 1.05)

ax.set_xlabel(ax.get_xlabel(), fontsize=14)

ax.set_ylabel(ax.get_ylabel(), fontsize=14)

plt.show()
plt.figure(figsize=(4, 5))

sns.countplot(train.Pclass)

plt.show()
plt.style.use('seaborn-poster')

plt.figure(figsize=(4, 5))

ax = sns.barplot(x = 'SibSp', y = 'Survived', data = train)

ax.set_title("Barplot Showing Survival Rate of Passengers having Siblings/Spouses", y = 1.05)

ax.set_xlabel(ax.get_xlabel(), fontsize=14)

ax.set_ylabel(ax.get_ylabel(), fontsize=14)

plt.show()
sns.set(font_scale = 1.3)

plt.style.use('seaborn-ticks')

plt.figure(figsize = (4, 5))

sns.countplot(train['SibSp'])

plt.show()
full.Parch.unique().tolist()
sns.set()

plt.style.use('fast')

plt.figure(figsize=(4, 5))

ax = sns.barplot(x = 'Parch', y = 'Survived', data = train)

ax.set_title("Barplot Showing Survival Rate of Passengers having Parents/Children", y = 1.05)

ax.set_xlabel(ax.get_xlabel(), fontsize=14)

ax.set_ylabel(ax.get_ylabel(), fontsize=14)

plt.show()
sns.set(font_scale = 1.2)

plt.figure(figsize = (4, 5))

sns.countplot(train['Parch'])

plt.show()
df_fare_pc = train.groupby('Pclass').mean().sort_values('Fare', ascending=False)['Fare']

df_fare_pc.head()
df_age_pc = train.groupby('Pclass')['Age'].mean()

df_age_pc.sort_values(ascending = False, inplace = True)

df_age_pc.head()
sns.set(font_scale = 2)

plt.style.use('seaborn-talk')

fig = plt.figure()

ax0 = fig.add_subplot(1, 2, 1) # add subplot 1 

ax1 = fig.add_subplot(1, 2, 2) # add subplot 2



# Subplot1 = Barplot of Pclass and Fare

df_fare_pc.plot(kind = 'barh', figsize=(20, 6), ax=ax0, color='crimson')

ax0.set_title("Barplot Of Average Fare of each Passenger Class", y = 1.04, fontsize = 17)

ax0.set_xlabel('Average Fare', fontsize = 17)

ax0.set_ylabel(ax0.get_ylabel(), fontsize = 17)

for index, value in enumerate(df_fare_pc):

    label = format(int(value), ',')

    ax0.annotate(label, color='white', xy = (value - 5, index - 0.05), fontsize = 15)



# Subplot2 = Barplot of Pclass and Age

df_age_pc.plot(kind = 'barh', figsize=(20, 6), ax=ax1, color='green')

ax1.set_title("Barplot Of Average Age in each Passenger Class", y = 1.04, fontsize = 17)

ax1.set_xlabel('Average Age', fontsize = 17)

ax1.set_ylabel(ax1.get_ylabel(), fontsize = 17)

for index, value in enumerate(df_age_pc):

    label = format(int(value), ',')

    ax1.annotate(label, color='white', xy = (value - 2.3, index - 0.05), fontsize = 15, )



plt.show()
plt.style.use('ggplot')

train.plot(kind='scatter', x='Age', y='Fare', figsize=(10, 6), color='darkblue')



plt.title('Scatter Plot of Fare vs Age')

plt.xlabel('Age of Passengers')

plt.ylabel('Fare')



plt.show()
train.info()
df_1 = train[train.Pclass == 1].loc[:, ['Fare', 'Survived']]

df_2 = train[train.Pclass == 2].loc[:, ['Fare', 'Survived']]

df_3 = train[train.Pclass == 3].loc[:, ['Fare', 'Survived']]
df_3.head()
df_1 = df_1.groupby('Survived')['Fare'].median()

df_2 = df_2.groupby('Survived')['Fare'].median()

df_3 = df_3.groupby('Survived')['Fare'].median()
df_4 = 100 * df_1 / df_1.sum()

df_5 = 100 * df_2 / df_2.sum()

df_6 = 100 * df_3 / df_3.sum()
sns.set(font_scale = 1.1)

plt.style.use('seaborn-whitegrid')

fig = plt.figure()

ax_0 = fig.add_subplot(1, 3, 1) # add subplot 1 

ax_1 = fig.add_subplot(1, 3, 2) # add subplot 2

ax_2 = fig.add_subplot(1, 3, 3) # add subplot 2



# Subplot1 = 1st class passengers

df_1.plot(kind = 'bar', ax=ax_0, color='crimson', sort_columns = True)

ax_0.set_title("1st Class Passengers", y = 1.04, fontsize = 17)

ax_0.set_xlabel('Survived', fontsize = 17)

ax_0.set_ylabel('Fare', fontsize = 17)



# Subplot2 = 2nd class passengers

df_2.plot(kind = 'bar', ax=ax_1, color='blue', sort_columns = True)

ax_1.set_title("2st Class Passengers", y = 1.04, fontsize = 17)

ax_1.set_xlabel('Survived', fontsize = 17)



# Subplot2 = 3rd class passengers

df_3.plot(kind = 'bar', ax=ax_2, color='green', sort_columns = True)

ax_2.set_title("3rd Class Passengers", y = 1.04, fontsize = 17)

ax_2.set_xlabel('Survived', fontsize = 17)

 

print("\t\t\tAmount of Fare Paid by each Passenger in each Class")    

plt.show()
sns.set(font_scale = 1.1)

plt.style.use('seaborn-whitegrid')

fig = plt.figure()

ax_0 = fig.add_subplot(1, 3, 1) # add subplot 1 

ax_1 = fig.add_subplot(1, 3, 2) # add subplot 2

ax_2 = fig.add_subplot(1, 3, 3) # add subplot 2



# Subplot1 = 1st class passengers

df_4.plot(kind='pie',

          figsize=(15, 6),

          autopct='%1.1f%%', 

          startangle=90,    

          shadow=True,       

          labels=None,         

          pctdistance=0.5,     

          colors=['gold', 'yellowgreen'],  

          explode=[0.1, 0],

          ax = ax_0,

          sharex = True,

          legend = True

          )

ax_0.set_title("1st Class Passengers", y = 1.04, fontsize = 17)

ax_0.axis('equal')



# Subplot2 = 2nd class passengers

df_5.plot(kind='pie',

          figsize=(15, 6),

          autopct='%1.1f%%', 

          startangle=90,    

          shadow=True,       

          labels=None,         

          pctdistance=0.5,     

          colors=['gold', 'yellowgreen'],  

          explode=[0.1, 0],

          ax = ax_1,

          sharex = True,

          legend = True

          )

ax_1.set_title("2nd Class Passengers", y = 1.04, fontsize = 17)

ax_1.axis('equal')



# Subplot2 = 3rd class passengers

df_6.plot(kind='pie',

          figsize=(15, 6),

          autopct='%1.1f%%', 

          startangle=90,    

          shadow=True,       

          labels=None,         

          pctdistance=0.5,     

          colors=['gold', 'yellowgreen'],  

          explode=[0.1, 0],

          ax = ax_2,

          sharex = True,

          legend = True

          )

ax_2.set_title("3rd Class Passengers", y = 1.04, fontsize = 17)

ax_2.axis('equal')



print("\t\t\t% of Fare paid by Passengers who survived in each class")



plt.show()
df_7 = full[full.Pclass == 1].groupby('Survived')['Fare'].count()

df_8 = full[full.Pclass == 2].groupby('Survived')['Fare'].count()

df_9 = full[full.Pclass == 3].groupby('Survived')['Fare'].count()
sns.set(font_scale = 1.1)

plt.style.use('seaborn-whitegrid')

fig = plt.figure()

ax_0 = fig.add_subplot(1, 3, 1) # add subplot 1 

ax_1 = fig.add_subplot(1, 3, 2) # add subplot 2

ax_2 = fig.add_subplot(1, 3, 3) # add subplot 2



# Subplot1 = 1st class passengers

df_7.plot(kind = 'bar', ax=ax_0, color='crimson', sort_columns = True)

ax_0.set_title("1st Class Passengers", y = 1.04, fontsize = 17)

ax_0.set_xlabel('Survived', fontsize = 17)

ax_0.set_ylabel('Count', fontsize = 17)



# Subplot2 = 2nd class passengers

df_8.plot(kind = 'bar', ax=ax_1, color='blue', sort_columns = True)

ax_1.set_title("2st Class Passengers", y = 1.04, fontsize = 17)

ax_1.set_xlabel('Survived', fontsize = 17)



# Subplot2 = 3rd class passengers

df_9.plot(kind = 'bar', ax=ax_2, color='green', sort_columns = True)

ax_2.set_title("3rd Class Passengers", y = 1.04, fontsize = 17)

ax_2.set_xlabel('Survived', fontsize = 17)

 

print("\t\t\tNumber of Passengers who paid fare in each class")    

plt.show()
count, bins = np.histogram(train['Age'])

count, bins
for i in range(train.shape[0]):

    if train.loc[i, 'Age'] <= 8:

        train.loc[i, 'Age Binned'] = '0 - 8'

    elif train.loc[i, 'Age'] <= 16:

        train.loc[i, 'Age Binned'] = '9 - 16'

    elif train.loc[i, 'Age'] <= 24:

        train.loc[i, 'Age Binned'] = '17 - 24'

    elif train.loc[i, 'Age'] <= 32:

        train.loc[i, 'Age Binned'] = '25 - 32'

    elif train.loc[i, 'Age'] <= 40:

        train.loc[i, 'Age Binned'] = '33 - 40'

    elif train.loc[i, 'Age'] <= 48:

        train.loc[i, 'Age Binned'] = '41 - 48'

    elif train.loc[i, 'Age'] <= 56:

        train.loc[i, 'Age Binned'] = '49 - 56'

    elif train.loc[i, 'Age'] <= 64:

        train.loc[i, 'Age Binned'] = '57 - 64'

    elif train.loc[i, 'Age'] <= 72:

        train.loc[i, 'Age Binned'] = '65 - 72'

    elif train.loc[i, 'Age'] <= 80:

        train.loc[i, 'Age Binned'] = '73 - 80'

    else:

        train.loc[i, 'Age Binned'] = '80+'
for i in range(test.shape[0]):

    if test.loc[i, 'Age'] <= 8:

        test.loc[i, 'Age Binned'] = '0 - 8'

    elif test.loc[i, 'Age'] <= 16:

        test.loc[i, 'Age Binned'] = '9 - 16'

    elif test.loc[i, 'Age'] <= 24:

        test.loc[i, 'Age Binned'] = '17 - 24'

    elif test.loc[i, 'Age'] <= 32:

        test.loc[i, 'Age Binned'] = '25 - 32'

    elif test.loc[i, 'Age'] <= 40:

        test.loc[i, 'Age Binned'] = '33 - 40'

    elif test.loc[i, 'Age'] <= 48:

        test.loc[i, 'Age Binned'] = '41 - 48'

    elif test.loc[i, 'Age'] <= 56:

        test.loc[i, 'Age Binned'] = '49 - 56'

    elif test.loc[i, 'Age'] <= 64:

        test.loc[i, 'Age Binned'] = '57 - 64'

    elif test.loc[i, 'Age'] <= 72:

        test.loc[i, 'Age Binned'] = '65 - 72'

    elif test.loc[i, 'Age'] <= 80:

        test.loc[i, 'Age Binned'] = '73 - 80'

    else:

        test.loc[i, 'Age Binned'] = '80+'
train.head()
for i in range(train.shape[0]):

    if train.loc[i, 'Fare'] <= 51:

        train.loc[i, 'Fare Binned'] = '0 - 51'

    elif train.loc[i, 'Fare'] <= 103:

        train.loc[i, 'Fare Binned'] = '52 - 103'

    elif train.loc[i, 'Fare'] <= 154:

        train.loc[i, 'Fare Binned'] = '104 - 154'

    elif train.loc[i, 'Fare'] <= 205:

        train.loc[i, 'Fare Binned'] = '155 - 205'

    elif train.loc[i, 'Fare'] <= 256:

        train.loc[i, 'Fare Binned'] = '206 - 256'

    elif train.loc[i, 'Fare'] <= 307:

        train.loc[i, 'Fare Binned'] = '257 - 307'

    elif train.loc[i, 'Fare'] <= 359:

        train.loc[i, 'Fare Binned'] = '308 - 359'

    elif train.loc[i, 'Fare'] <= 410:

        train.loc[i, 'Fare Binned'] = '360 - 410'

    elif train.loc[i, 'Fare'] <= 461:

        train.loc[i, 'Fare Binned'] = '411 - 461'

    elif train.loc[i, 'Fare'] <= 512:

        train.loc[i, 'Fare Binned'] = '462 - 512'

    else:

        train.loc[i, 'Fare Binned'] = '512+'

        

for i in range(test.shape[0]):

    if test.loc[i, 'Fare'] <= 51:

        test.loc[i, 'Fare Binned'] = '0 - 51'

    elif test.loc[i,'Fare'] <= 103:

        test.loc[i, 'Fare Binned'] = '52 - 103'

    elif test.loc[i, 'Fare'] <= 154:

        test.loc[i, 'Fare Binned'] = '104 - 154'

    elif test.loc[i, 'Fare'] <= 205:

        test.loc[i, 'Fare Binned'] = '155 - 205'

    elif test.loc[i, 'Fare'] <= 256:

        test.loc[i, 'Fare Binned'] = '206 - 256'

    elif test.loc[i, 'Fare'] <= 307:

        test.loc[i, 'Fare Binned'] = '257 - 307'

    elif test.loc[i, 'Fare'] <= 359:

        test.loc[i, 'Fare Binned'] = '308 - 359'

    elif test.loc[i, 'Fare'] <= 410:

        test.loc[i, 'Fare Binned'] = '360 - 410'

    elif test.loc[i, 'Fare'] <= 461:

        test.loc[i, 'Fare Binned'] = '411 - 461'

    elif test.loc[i, 'Fare'] <= 512:

        test.loc[i, 'Fare Binned'] = '462 - 512'

    else:

        test.loc[i, 'Fare Binned'] = '512+'
train.head()
sns.set(font_scale = 1.2)

plt.style.use(['ggplot', 'bmh'])



fig, ax = plt.subplots(1, 2, figsize=(12, 7))

sns.barplot(x = 'Age Binned', y = 'Survived', data = train, ax = ax[0])

ax[0].set_title("Barplot Showing Survival Rate w.r.t Different Age Groups", y = 1.05)

ax[0].set_xlabel("Age Groups", fontsize=16)

ax[0].set_ylabel("Survival Rate", fontsize=16)

ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)



sns.barplot(x = 'Fare Binned', y = 'Survived', data = train, ax = ax[1])

ax[1].set_title("Barplot Showing Survival Rate w.r.t Different Fare Types", y = 1.05)

ax[1].set_xlabel("Fare Types", fontsize=16)

ax[1].set_ylabel("Survival Rate", fontsize=16)

ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)

plt.show()
train['FamilySize'] = train.Parch + train.SibSp + 1

test['FamilySize'] = test.Parch + test.SibSp + 1

train.drop(columns = ['Age', 'Fare'], axis = 1, inplace = True)
train.set_index('PassengerId', inplace = True)

test.set_index('PassengerId', inplace = True)

train.head()
train.FamilySize = train.FamilySize.astype(str)

train.info()
test.drop(columns = ['Age', 'Fare'], axis = 1, inplace = True)

train.FamilySize = train.FamilySize.astype(str)
train.info()
train.tail()
# Extract the salutation!

train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
test_sal = list(set(test.Title.unique().tolist()) - set(train.Title.unique().tolist()))

test_sal
train_sal = set(train.Title.unique().tolist()) - set(test.Title.unique().tolist())
test.Title.unique().tolist()
train.Title.unique().tolist()
train['Title'].value_counts(normalize = True) * 100
test.Title.value_counts(normalize = True) * 100
# Bundle rare salutations: 'Other' category

train['Title'] = train['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 

                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')
# Bundle rare salutations: 'Other' category

test['Title'] = test['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 

                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')
train.Title.value_counts(normalize=True) * 100
test.Title.value_counts(normalize=True) * 100
test.isnull().sum()
train.head()
# Embarked, Sex: OneHotEncoder

# Cabin, Title: TargetEncoder

# Name, Ticket: To be droped

train.drop(columns=['Name', 'Ticket'], inplace=True)

test.drop(columns=['Name', 'Ticket'], inplace=True)

train.head()
# We will also drop Parch and SibSp

train.drop(columns=['Parch', 'SibSp'], inplace=True)

test.drop(columns=['Parch', 'SibSp'], inplace=True)
test.head()
y = pd.DataFrame(train.Survived, index = train.index, columns = ['Survived'])

y.head()
train.drop(columns='Survived', inplace = True)
train.Pclass = train.Pclass.astype(str)

test.Pclass = test.Pclass.astype(str)

train.info()
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.20, random_state = 42)

y_train.head()
from category_encoders.one_hot import OneHotEncoder



# First doing One Hot Encoding

OH_encoder = OneHotEncoder(handle_unknown='ignore', cols = ['Embarked', 'Sex'], use_cat_names = True)

num_encoder = TargetEncoder(handle_unknown = 'value', cols=['Cabin', 'Title', 'Age Binned', 'Fare Binned', 'FamilySize', 'Pclass'])



pipeline = Pipeline([('one_hot', OH_encoder),

                    ('target', num_encoder)])
num_X_train = pd.DataFrame(pipeline.fit_transform(X_train, y_train))

num_X_val = pd.DataFrame(pipeline.transform(X_test))

num_X_test = pd.DataFrame(pipeline.transform(test))

num_X_train.drop(columns = ['Embarked_S', 'Sex_male'], axis = 1, inplace = True)

num_X_val.drop(columns = ['Embarked_S', 'Sex_male'], axis = 1, inplace = True)

num_X_test.drop(columns = ['Embarked_S', 'Sex_male'], axis = 1, inplace = True)

num_X_test.head()
num_X_test.isnull().sum()
num_X_val.isnull().sum()
num_X_train.describe()
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import train_test_split



# Keep 5 features

selector = SelectKBest(f_classif, k=6)



X_new = selector.fit_transform(num_X_train, y_train)

X_new
# Get back the features we've kept, zero out all other features

selected_features_train = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=num_X_train.index, 

                                 columns=num_X_train.columns)

selected_features_train.head()
# Dropped columns have values of all 0s, so var is 0, drop them

selected_columns = selected_features_train.columns[selected_features_train.var() != 0]



# Get the valid dataset with the selected features.

num_X_val[selected_columns].head()

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel



# Set the regularization parameter C=1

logistic = LogisticRegression(C=1, penalty="l1", random_state=7, solver = 'liblinear').fit(num_X_train,y_train) # In later models, it was found out Age Binned and Fare Binned didn't contribute much to model building. So, we are dropping it here to avoid data leakage.

model = SelectFromModel(logistic, prefit=True)



X_new = model.transform(num_X_train)

X_new
# Get back the kept features as a DataFrame with dropped columns as all 0s

selected_features_train2 = pd.DataFrame(model.inverse_transform(X_new), 

                                 index=num_X_train.index,

                                 columns=num_X_train.columns)



# Dropped columns have values of all 0s, keep other columns 

selected_columns2 = selected_features_train2.columns[selected_features_train2.var() != 0]



num_X_val[selected_columns2].head()
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 1000).fit(num_X_train, y_train)



model = SelectFromModel(rf, prefit=True)



X_new = model.transform(num_X_train)

X_new
# Get back the kept features as a DataFrame with dropped columns as all 0s

selected_features_train3 = pd.DataFrame(model.inverse_transform(X_new), 

                                 index=num_X_train.index,

                                 columns=num_X_train.columns)



# Dropped columns have values of all 0s, keep other columns 

selected_columns3 = selected_features_train3.columns[selected_features_train3.var() != 0]



num_X_val[selected_columns3].head()
rf = RandomForestClassifier(n_estimators = 1000)



rfecv = RFE(estimator=rf, step=1)

rfecv.fit(num_X_train, y_train)



print("Optimal number of features : %d" % rfecv.n_features_)
selected_columns4 = num_X_val.columns[rfecv.support_]



pd.DataFrame({'Ranking' : rfecv.ranking_}, index=num_X_val.columns)
import lightgbm as lgb

from sklearn import metrics



def train_model(X_train, y_train, X_test, y_test, feature_cols):

    dtrain = lgb.Dataset(X_train[feature_cols], label=y_train)

    dvalid = lgb.Dataset(X_test[feature_cols], label=y_test)



    param = {'num_leaves': 64, 'objective': 'binary', 

             'metric': 'auc', 'seed': 7}

    print("Training model!")

    bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid], 

                    early_stopping_rounds=10, verbose_eval=False)



    valid_pred = bst.predict(X_test[feature_cols])

    valid_score = metrics.roc_auc_score(y_test, valid_pred)

    print(f"Validation AUC score: {valid_score:.4f}")

    return bst
# Dataset1



_ = train_model(num_X_train, y_train, num_X_val, y_test, selected_columns)
# Dataset2



_ = train_model(num_X_train, y_train, num_X_val, y_test, selected_columns2)
# Dataset3



_ = train_model(num_X_train, y_train, num_X_val, y_test, selected_columns3)
# Dataset4



_ = train_model(num_X_train, y_train, num_X_val, y_test, selected_columns4)
param = {

    'min_samples_split': hp.uniform('min_samples_split', 0.1, 0.95),

    'max_depth': hp.quniform('max_depth', 2, 15, 2),

    'ccp_alpha': hp.uniform('ccp_alpha', 0.001,0.9), 

}



np.random.seed(0)

rstate = np.random.RandomState(42)

# Set up objective function

def objective(params):

    params = {'max_depth': int(params['max_depth']),

              'ccp_alpha': params['ccp_alpha'], 

              'min_samples_split' : params['min_samples_split']

              }

    tree_clf = DecisionTreeClassifier(**params) 

    tree_clf.fit(num_X_train[selected_columns], y_train.Survived)

    y_pred = tree_clf.predict(num_X_val[selected_columns])

    score = accuracy_score(y_test.Survived, y_pred)

    loss = 1 - score

    return loss



# Run the algorithm

trials = Trials()

best1 = fmin(fn=objective, space=param, max_evals=1000, rstate=rstate, algo=tpe.suggest, trials = trials)

print(best1)
tree = DecisionTreeClassifier(min_samples_split = best1['min_samples_split'],

                             max_depth = best1['max_depth'],

                             ccp_alpha = best1['ccp_alpha'])

tree.fit(num_X_train[selected_columns], y_train.Survived)

y_tr = tree.predict(num_X_train[selected_columns])

tr_score = accuracy_score(y_train.Survived.to_numpy(), y_tr)

y_val = tree.predict(num_X_val[selected_columns])

val_score = accuracy_score(y_test.Survived.to_numpy(), y_val)

print('Validation Results:\n Training Accuracy =', tr_score, ', Validation Score =', val_score)
y_pred = tree.predict(num_X_test[selected_columns])

predictions = pd.DataFrame({'Survived': y_pred}, index = num_X_test.index)

predictions.Survived = predictions.Survived.astype('int64')

predictions.info()
predictions.to_csv('submission1.csv')
pd.DataFrame(tree.feature_importances_, index = selected_columns).plot.bar(legend=False)

plt.show()
param = {

    'n_neighbors': hp.quniform('n_neighbors', 2, 15, 2),

    'weights' : hp.choice('weights', ('uniform', 'distance'))

}



np.random.seed(0)

rstate = np.random.RandomState(42)

# Set up objective function

def objective(params):

    params = {'n_neighbors': int(params['n_neighbors']),

              'weights': params['weights']

              }

    knn = KNeighborsClassifier(**params) 

    knn.fit(num_X_train[selected_columns], y_train.Survived)

    y_pred = knn.predict(num_X_val[selected_columns])

    score = accuracy_score(y_test.Survived, y_pred)

    loss = 1 - score

    return loss



# Run the algorithm

trials = Trials()

best2 = fmin(fn=objective, space=param, max_evals=1000, rstate=rstate, algo=tpe.suggest, trials = trials)

print(best2)
w = ('uniform', 'distance')

knn = KNeighborsClassifier(n_neighbors = int(best2['n_neighbors']), weights = w[best2['weights']])

knn.fit(num_X_train[selected_columns], y_train.Survived)
y_tr = knn.predict(num_X_train[selected_columns])

tr_score = accuracy_score(y_train.Survived.to_numpy(), y_tr)

y_val = knn.predict(num_X_val[selected_columns])

val_score = accuracy_score(y_test.Survived.to_numpy(), y_val)

print('Validation Results:\n Training Accuracy =', tr_score, ', Validation Score =', val_score)
y_pred = knn.predict(num_X_test[selected_columns])

predictions = pd.DataFrame({'Survived': y_pred}, index = num_X_test.index)

predictions.Survived = predictions.Survived.astype('int64')

predictions.info()
predictions.to_csv('submission2.csv')
param = {

    'degree': hp.quniform('degree', 1, 20, 1),

    'C': hp.quniform('C', 1, 1000, 1),

    'gamma': hp.uniform('gamma', 0.0001, 0.9),

}



np.random.seed(0)

rstate = np.random.RandomState(42)

# Set up objective function

def objective(params):

    params = {'degree' : int(params['degree']),

              'C' : int(params['C']),

              'gamma' : params['gamma'],

              }

    svm = SVC(**params) 

    svm.fit(num_X_train[selected_columns], y_train.Survived)

    y_pred = svm.predict(num_X_val[selected_columns])

    score = accuracy_score(y_test.Survived, y_pred)

    loss = 1 - score

    return loss



# Run the algorithm

trials = Trials()

best3 = fmin(fn=objective, space=param, max_evals=1000, rstate=rstate, algo=tpe.suggest, trials = trials)

print(best3)
svm = SVC(C = int(best3['C']), degree = int(best3['degree']), gamma = best3['gamma'])

svm.fit(num_X_train[selected_columns], y_train.Survived)
y_tr = svm.predict(num_X_train[selected_columns])

tr_score = accuracy_score(y_train.Survived.to_numpy(), y_tr)

y_val = svm.predict(num_X_val[selected_columns])

val_score = accuracy_score(y_test.Survived.to_numpy(), y_val)

print('Validation Results:\n Training Accuracy =', tr_score, ', Validation Score =', val_score)
y_pred = svm.predict(num_X_test[selected_columns])

predictions = pd.DataFrame({'Survived': y_pred}, index = num_X_test.index)

predictions.Survived = predictions.Survived.astype('int64')

predictions.info()
predictions.to_csv('submission3.csv')
naive = GaussianNB()

naive.fit(num_X_train[selected_columns], y_train.Survived)

y_tr = naive.predict(num_X_train[selected_columns])

tr_score = accuracy_score(y_train.Survived.to_numpy(), y_tr)

y_val = naive.predict(num_X_val[selected_columns])

val_score = accuracy_score(y_test.Survived.to_numpy(), y_val)

print('Validation Results:\n Training Accuracy =', tr_score, ', Validation Score =', val_score)
y_pred = naive.predict(num_X_test[selected_columns])

predictions = pd.DataFrame({'Survived': y_pred}, index = num_X_test.index)

predictions.Survived = predictions.Survived.astype('int64')

predictions.info()
predictions.to_csv('submission4.csv')
# Define get_stacking():

def get_stacking():

    

	# Create an empty list for the base models called layer1

    layer1 = []

    w = ('uniform', 'distance')

  # Append tuple with classifier name and instantiations (no arguments) for DecisionTreeClassifier, KNeighborsClassifier, SVC, and GaussianNB base models

  # Hint: layer1.append(('ModelName', Classifier()))

    layer1.append(('DT', DecisionTreeClassifier(min_samples_split = best1['min_samples_split'],

                             max_depth = best1['max_depth'],

                             ccp_alpha = best1['ccp_alpha'])))

    layer1.append(('KNN', KNeighborsClassifier(n_neighbors = int(best2['n_neighbors']), weights = w[best2['weights']])))

    layer1.append(('SVM', SVC(C = int(best3['C']), degree = int(best3['degree']), gamma = best3['gamma'])))

    layer1.append(('Bayes', GaussianNB()))



  # Instantiate Logistic Regression as meta learner model called layer2

    layer2 = LogisticRegression()



	# Define StackingClassifier() called model passing layer1 model list and meta learner with 5 cross-validations

    model = StackingClassifier(estimators = layer1, final_estimator = layer2, cv = 5)



  # return model

    return model
# Define get_models():

def get_models():



  # Create empty dictionary called models

    models = dict()



  # Add key:value pairs to dictionary with key as ModelName and value as instantiations (no arguments) for DecisionTreeClassifier, KNeighborsClassifier, SVC, and GaussianNB base models

  # Hint: models['ModelName'] = Classifier()

    models['DT'] = DecisionTreeClassifier(min_samples_split = best1['min_samples_split'],

                             max_depth = best1['max_depth'],

                             ccp_alpha = best1['ccp_alpha'])

    models['KNN'] = KNeighborsClassifier(n_neighbors = int(best2['n_neighbors']), weights = w[best2['weights']])

    models['SVM'] = SVC(C = int(best3['C']), degree = int(best3['degree']), gamma = best3['gamma'])

    models['Bayes'] = GaussianNB()

    

  # Add key:value pair to dictionary with key called Stacking and value that calls get_stacking() custom function

    models['Stacking'] = get_stacking()



  # return dictionary

    return models
# Define evaluate_model:

def evaluate_model(model):

    model.fit(num_X_train[selected_columns], y_train.Survived)

    y_val = model.predict(num_X_val[selected_columns])

    val_score = accuracy_score(y_test.Survived.to_numpy(), y_val)



  # return scores

    return val_score
# Assign get_models() to a variable called models

models = get_models()
# Evaluate the models and store results

# Create an empty list for the results

results = list()



# Create an empty list for the model names

names = list()



# Create a for loop that iterates over each name, model in models dictionary 

for name, model in models.items():



	# Call evaluate_model(model) and assign it to variable called scores

	scores = evaluate_model(model)

 

  # Append output from scores to the results list

	results.append(scores)

 

  # Append name to the names list

	names.append(name)

 

  # Print name, mean and standard deviation of scores:

	print('>%s %.3f' % (name, scores))



# Plot model performance for comparison using names for x and results for y and setting showmeans to True

sns.barplot(x=names, y=results)
n = len(selected_columns)



# Instantiate a Sequential Model

model = Sequential()



# Add input and hidden layer

model.add(Dense(10, input_shape=(n, ), activation='tanh'))

model.add(Dense(1, activation='sigmoid'))
# Compile model

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])



# Train model

model.fit(num_X_train[selected_columns], y_train.Survived, epochs=5000)



print('Training Loss:', model.evaluate(num_X_train[selected_columns], y_train.Survived))
print('Test Loss:', model.evaluate(num_X_val[selected_columns], y_val))
y_pred = model.predict_classes(num_X_test[selected_columns])

predictions = pd.DataFrame({'Survived': list(y_pred)}, index = num_X_test.index)

predictions.Survived = predictions.Survived.astype('int64')

predictions.info()
predictions.to_csv('submission2.csv')