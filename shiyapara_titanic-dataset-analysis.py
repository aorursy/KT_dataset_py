# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use(style = 'ggplot')

import seaborn as sns

sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))  

        



# Any results you write to the current directory are saved as output
train=pd.read_csv("../input/titanic/train.csv" )

test=pd.read_csv("../input/titanic/test.csv" )

train.head()
train.info()
train.isnull().sum()
train.describe(include="all")
## To find out duplicates in the dataset:

# There are 210 duplicate tickets and 57 cabins shared.

train.describe(include=['O'])
train.drop (['PassengerId','Ticket', 'Fare','Cabin'],axis=1, inplace=True)

train.head(5)
corr= train.corr()**2

corr.Survived.sort_values(ascending= False)
## heatmeap to see the correlation between features. 

# Generate a mask for the upper triangle (taken from seaborn example gallery)

import numpy as np

mask = np.zeros_like(train.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.set_style('whitegrid')

plt.subplots(figsize = (15,12))

sns.heatmap(train.corr(), 

            annot=True,

            mask = mask,

            cmap = 'RdBu', ## in order to reverse the bar replace "RdBu" with "RdBu_r"

            linewidths=.9, 

            linecolor='white',

            fmt='.2g',

            center = 0,

            square=True)

plt.title("Correlations Among Features", y = 1.03,fontsize = 20, pad = 40);
for dataset in train:

    age_avg = train['Age'].mean()

    age_std = train['Age'].std()

    age_null = train ['Age'].isnull().sum()

    

    random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null)

    train['Age'][np.isnan(train['Age'])] = random_list

    train['Age'] = train['Age'].astype(int)

    
age_cat = pd.cut(train.Age, bins = [0,2,13,17,60,99], labels= ['Infants','children','Young Adults','Adults', 'Elderly'])



age_cat 
train.insert(5, "AgeCat", age_cat,)

train.head()
# Southampton has higest entry. Therefore, using it's value "S" to fill in the gaps.

train['Embarked'].fillna('S', inplace = True)
# To verify that all null values are filled.

train.Embarked.value_counts()
## Name:

train['Title'] = train.Name.str.extract('([A-Za-z]+)\.')



for data in train:

    train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Other')

    train['Title'] = train['Title'].replace('Mlle','Miss')

    train['Title'] = train['Title'].replace('Ms','Miss')

    train['Title'] = train['Title'].replace('Mme','Mrs')

    

print(pd.crosstab(train['Title'],train['Sex']))
train.head(100)
# First calculate number of survived Vs Dead



Survived = train[train['Survived']==1]

dead = train[train['Survived']==0]

print("Survived:%i (%.1f%%)"%(len(Survived), float(len(Survived))/len(train)*100.0))

print("dead:%i (%.1f%%)"%(len(dead), float(len(dead))/len(train)*100.0))

print("Total:%i"%len(train))
train.Pclass.value_counts()
train.groupby('Pclass').Survived.value_counts()
mean_Pclass = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean()

print(mean_Pclass)
sns.barplot(x='Pclass', y='Survived', data=mean_Pclass)
train.Sex.value_counts()
train.groupby('Sex').Survived.value_counts()

mean_sex = train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()

print(mean_sex)
sns.barplot(x ='Sex', y ='Survived', data = train)
survived_gender = pd.crosstab(train['Survived'], train['Sex'])

survived_gender
# Percentage of passengers sirvived, female vs male



(train[['Sex','Survived']].groupby(['Sex']).mean()*100).round(2)
# Plotting the survival by gender



survived_gender.plot(kind='bar', stacked=True, rot = 0, figsize=(10, 7));
train.AgeCat.value_counts()
train.groupby('AgeCat').Survived.value_counts()
mean_agecat = train[['AgeCat', 'Survived']].groupby(['AgeCat'], as_index = False).mean()

print(mean_agecat)
sns.barplot( x = 'AgeCat', y = 'Survived', data = train)
agecat_survived = train.pivot_table('Survived', columns = 'Sex', index = 'AgeCat', aggfunc="sum")

agecat_survived
agecat_survived.plot(title = " Total Survival by AgeCat and Sex", figsize=(6,4));
# to understand the relationship between passenger class, sex and survival rate.



t_combined = pd.crosstab(train['Pclass'],train['Sex'])

print(t_combined)
t_combined = sns.catplot(x="Sex", y="Survived", col="Pclass",data=train, saturation=.5,

                kind="bar", ci=None, aspect=.6)

(t_combined .set_axis_labels("", "Survival Rate")

  .set_xticklabels(["Male", "Female"])

  .set_titles("{col_name} {col_var}")

  .set(ylim=(0, 1))

  .despine(left=True)) 
train.Embarked.value_counts()
train.groupby('Embarked').Survived.value_counts()
train[['Embarked','Survived']].groupby(['Embarked'], as_index = False).mean()
sns.barplot(x ='Embarked', y = 'Survived', data = train)
p_combined = pd.crosstab(train['Pclass'],train['Embarked'])

print(p_combined)
p_combined = sns.catplot(x="Embarked", y="Survived", col="Pclass",data=train, saturation=.5,

                kind="bar", ci=None, aspect=.6)

(t_combined .set_axis_labels("", "Survival Rate")

  .set_xticklabels(["C", "Q", "S"])

  .set_titles("{col_name} {col_var}")

  .set(ylim=(0, 1))

  .despine(left=True)) 
all_combined = sns.catplot (x="Age", y="Embarked",

                hue="Sex", row="Pclass",

                data=train[train.Embarked.notnull()],

                orient="h", height=4, aspect=3, palette="Set2",

                kind="box")