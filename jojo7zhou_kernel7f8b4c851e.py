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
# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

sns.set(style="darkgrid")

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import StratifiedKFold



import string

import warnings

warnings.filterwarnings('ignore')



SEED = 42
#Read The Data

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train
dfs = [train , test]

def display_missing(df):    

    for col in df.columns.tolist():          

        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))

    print('\n')



    

for df in dfs:

    print('{}'.format(df['Name']))

    display_missing(df)



    

pal = {1:"seagreen", 0:"gray"}

sns.set(style="darkgrid")

plt.subplots(figsize = (15,8))

ax = sns.countplot(x = "Sex", 

                   hue="Survived",

                   data = train, 

                   linewidth=4, 

                   palette = pal

)



## Fixing title, xlabel and ylabel

plt.title("Passenger Gender Distribution - Survived vs Not-survived", fontsize = 25, pad=40)

plt.xlabel("Sex", fontsize = 15);

plt.ylabel("# of Passenger Survived", fontsize = 15)



## Fixing xticks

#labels = ['Female', 'Male']

#plt.xticks(sorted(train.Sex.unique()), labels)



## Fixing legends

leg = ax.get_legend()

leg.set_title("Survived")

legs = leg.texts

legs[0].set_text("No")

legs[1].set_text("Yes")

plt.show()

count = [train[train['Sex'] == 'female']['Sex'].count(),train[train['Sex'] == 'male']['Sex'].count()]

plt.pie(count,labels = ('female','male'),autopct ='%1.1f%%', explode = (0,  0.05))
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

temp = train[['Pclass', 'Survived', 'PassengerId']].groupby(['Pclass', 'Survived']).count().reset_index()

temp_df = pd.pivot_table(temp, values = 'PassengerId', index = 'Pclass',columns = 'Survived')

names = ['No', 'Yes']

temp_df.columns = names

r = [0,1,2]

totals = [i+j for i, j in zip(temp_df['No'], temp_df['Yes'])]

No_s = [i / j * 100 for i,j in zip(temp_df['No'], totals)]

Yes_s = [i / j * 100 for i,j in zip(temp_df['Yes'], totals)]

## Plotting

plt.subplots(figsize = (15,10))

barWidth = 0.60

names = ('Upper', 'Middle', 'Lower')

# Create green Bars

plt.bar(r, No_s, color='Red', edgecolor='white', width=barWidth)

# Create orange Bars

plt.bar(r, Yes_s, bottom=No_s, color='Green', edgecolor='white', width=barWidth)



 

# Custom x axis

plt.xticks(r, names)

plt.xlabel("Pclass")

plt.ylabel('Percentage')

 

# Show graphic

plt.show()
plt.subplots(figsize = (15,10))

sns.barplot(x = "Pclass", 

            y = "Survived", 

            data=train, 

            linewidth=6,

            capsize = .05,

            errcolor='blue',

            errwidth = 3

            



           )

plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25, pad=40)

plt.xlabel("Socio-Economic class", fontsize = 15);

plt.ylabel("% of Passenger Survived", fontsize = 15);

names = ['Upper', 'Middle', 'Lower']

#val = sorted(train.Pclass.unique())

val = [0,1,2] ## this is just a temporary trick to get the label right. 

plt.xticks(val, names);
a = [train[train['Pclass'] == 1]['Pclass'].count(),train[train['Pclass'] == 2]['Pclass'].count(),train[train['Pclass'] == 3]['Pclass'].count()]

plt.pie(a,labels = ('Upper','Middle','Lower'),autopct ='%1.1f%%', explode = (0, 0, 0.05))



#print percentage of people by Pclass that survived

print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
age_by_pclass_sex = train.groupby(['Sex', 'Pclass']).median()['Age']



for pclass in range(1, 4):

    for sex in ['female', 'male']:

        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))

print('Median age of all passengers: {}'.format(train['Age'].median()))



# Filling the missing values in Age with the medians of Sex and Pclass groups

train['Age'] = train.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
# grid = sns.FacetGrid(train_df, col='Embarked')

grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
test[test['Embarked'].isnull().values==True]
train[train['Embarked'].isnull().values==True]
# The most common location

freq_port = train.Embarked.dropna().mode()[0]

freq_port
all = [train, test]

for dataset in all:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

dataset

    

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
med_fare = train.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]

# Filling the missing value in Fare with the median Fare of 3rd class alone passenger

train['Fare'] = train['Fare'].fillna(med_fare)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.factorplot(x =  "SibSp", y = "Survived", data = train,kind = "point",size = 8)

plt.title('Factorplot of Sibilings/Spouses survived', fontsize = 25)

plt.subplots_adjust(top=0.85)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
## dropping the three outliers where Fare is over $500 

train = train[train.Fare < 500]

## factor plot

sns.factorplot(x = "Parch", y = "Survived", data = train,kind = "point",size = 8)

plt.title("Factorplot of Parents/Children survived", fontsize = 25)

plt.subplots_adjust(top=0.85)
# mean value

train[train['Survived']== 1]['Survived'].count()/train['Survived'].count()
survived = train['Survived'].value_counts()[1]

not_survived = train['Survived'].value_counts()[0]

survived_per = survived / train.shape[0] * 100

not_survived_per = not_survived / train.shape[0] * 100



print('{} of {} passengers survived and it is the {:.2f}% of the training set.'.format(survived, train.shape[0], survived_per))

print('{} of {} passengers didnt survive and it is the {:.2f}% of the training set.'.format(not_survived, train.shape[0], not_survived_per))



plt.figure(figsize=(10, 8))

sns.countplot(train['Survived'])



plt.xlabel('Survival', size=15, labelpad=15)

plt.ylabel('Passenger Count', size=15, labelpad=15)

plt.xticks((0, 1), ['Not Survived ({0:.2f}%)'.format(not_survived_per), 'Survived ({0:.2f}%)'.format(survived_per)])

plt.tick_params(axis='x', labelsize=13)

plt.tick_params(axis='y', labelsize=13)



plt.title('Training Set Survival Distribution', size=15, y=1.05)



plt.show()
train['name_length'] = [len(i) for i in train.Name]

test['name_length'] = [len(i) for i in test.Name]



def name_length_group(size):

    a = ''

    if (size <=20):

        a = 'short'

    elif (size <=35):

        a = 'medium'

    elif (size <=45):

        a = 'good'

    else:

        a = 'long'

    return a





train['nLength_group'] = train['name_length'].map(name_length_group)

test['nLength_group'] = test['name_length'].map(name_length_group)
## get the title from the name

train["title"] = [i.split('.')[0] for i in train.Name]

train["title"] = [i.split(',')[1] for i in train.title]
print(train.title.unique())
## Let's fix that

train.title = train.title.apply(lambda x: x.strip())
test['title'] = [i.split('.')[0].split(',')[1].strip() for i in test.Name]
## Let's replace some of the rare values with the keyword 'rare' and other word choice of our own. 

## train Data

train["title"] = [i.replace('Ms', 'Miss') for i in train.title]

train["title"] = [i.replace('Mlle', 'Miss') for i in train.title]

train["title"] = [i.replace('Mme', 'Mrs') for i in train.title]

train["title"] = [i.replace('Dr', 'rare') for i in train.title]

train["title"] = [i.replace('Col', 'rare') for i in train.title]

train["title"] = [i.replace('Major', 'rare') for i in train.title]

train["title"] = [i.replace('Don', 'rare') for i in train.title]

train["title"] = [i.replace('Jonkheer', 'rare') for i in train.title]

train["title"] = [i.replace('Sir', 'rare') for i in train.title]

train["title"] = [i.replace('Lady', 'rare') for i in train.title]

train["title"] = [i.replace('Capt', 'rare') for i in train.title]

train["title"] = [i.replace('the Countess', 'rare') for i in train.title]

train["title"] = [i.replace('Rev', 'rare') for i in train.title]
## we are writing a function that can help us modify title column

def name_converted(feature):

    """

    This function helps modifying the title column

    """

    

    result = ''

    if feature in ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col', 'Rev', 'Dona', 'Dr']:

        result = 'rare'

    elif feature in ['Ms', 'Mlle']:

        result = 'Miss'

    elif feature == 'Mme':

        result = 'Mrs'

    else:

        result = feature

    return result



test.title = test.title.map(name_converted)

train.title = train.title.map(name_converted)
print(train.title.unique())

print(test.title.unique())
## Family_size seems like a good feature to create

train['family_size'] = train.SibSp + train.Parch+1

test['family_size'] = test.SibSp + test.Parch+1
## bin the family size. 

def family_group(size):

    """

    This funciton groups(loner, small, large) family based on family size

    """

    

    a = ''

    if (size <= 1):

        a = 'loner'

    elif (size <= 4):

        a = 'small'

    else:

        a = 'large'

    return a
## apply the family_group function in family_size

train['family_group'] = train['family_size'].map(family_group)

test['family_group'] = test['family_size'].map(family_group)
train['is_alone'] = [1 if i<2 else 0 for i in train.family_size]

test['is_alone'] = [1 if i<2 else 0 for i in test.family_size]
## Calculating fare based on family size. 

train['calculated_fare'] = train.Fare/train.family_size

test['calculated_fare'] = test.Fare/test.family_size
def fare_group(fare):

    """

    This function creates a fare group based on the fare provided

    """

    

    a= ''

    if fare <= 4:

        a = 'Very_low'

    elif fare <= 10:

        a = 'low'

    elif fare <= 20:

        a = 'mid'

    elif fare <= 45:

        a = 'high'

    else:

        a = "very_high"

    return a



train['fare_group'] = train['calculated_fare'].map(fare_group)

test['fare_group'] = test['calculated_fare'].map(fare_group)
train = pd.get_dummies(train, columns=['title',"Pclass", 'Embarked','nLength_group', 'family_group', 'fare_group'], drop_first=False)

test = pd.get_dummies(test, columns=['title',"Pclass",'Embarked','nLength_group', 'family_group', 'fare_group'], drop_first=False)

train.drop(['family_size','Name', 'Fare','name_length'], axis=1, inplace=True)

test.drop(['Name','family_size',"Fare",'name_length'], axis=1, inplace=True)

train
## rearranging the columns so that I can easily use the dataframe to predict the missing age values. 

train = pd.concat([train[["Survived", "Age", "Sex","SibSp","Parch"]], train.loc[:,"is_alone":]], axis=1)

test = pd.concat([test[["Age", "Sex"]], test.loc[:,"SibSp":]], axis=1)
def age_group_fun(age):

    """

    This function creates a bin for age

    """

    a = ''

    if age <= 1:

        a = 'infant'

    elif age <= 4: 

        a = 'toddler'

    elif age <= 13:

        a = 'child'

    elif age <= 18:

        a = 'teenager'

    elif age <= 35:

        a = 'Young_Adult'

    elif age <= 45:

        a = 'adult'

    elif age <= 55:

        a = 'middle_aged'

    elif age <= 65:

        a = 'senior_citizen'

    else:

        a = 'old'

    return a

## Applying "age_group_fun" function to the "Age" column.

train['age_group'] = train['Age'].map(age_group_fun)

test['age_group'] = test['Age'].map(age_group_fun)



## Creating dummies for "age_group" feature. 

train = pd.get_dummies(train,columns=['age_group'], drop_first=True)

test = pd.get_dummies(test,columns=['age_group'], drop_first=True);
train.sample(20)