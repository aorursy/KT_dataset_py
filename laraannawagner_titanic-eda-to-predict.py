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
import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import missingno as msno
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
# take look our data frame

train_df.head()
# let's look from bottom

train_df.tail()
#Check data type

train_df.dtypes
msno.bar(train_df,color="dodgerblue", sort="ascending", figsize=(10,5), fontsize=12)
#let's look at list if sum of null values

train_df.isnull().sum()
# we can look at our missing value with matris that is provide us to understand those missing values are rondom or not

msno.matrix(train_df)
# Drop cabin column

train_df = train_df.drop(["Cabin"], axis=1)
# Filling of Embarked Column

train_df[train_df['Embarked'].isna()] #Passengers travelling together



sub_embarked = train_df[(train_df['Fare'] > 79) & (train_df['Fare'] < 81) & (train_df['Pclass'] == 1)]

fill_mode = sub_embarked["Embarked"].mode()[0]





train_df = train_df.fillna({'Embarked': fill_mode})
# Filling of Age Coulmn

train_df["Title"] = train_df["Name"].apply(lambda x: x.split(",")[1].split(".")[0])



train_df['Age'] = train_df.groupby(['Title'])['Age'].apply(lambda x: x.fillna(x.median()))



train_df.head(10)
# Check to missing values again

train_df.info()
pd.crosstab([train_df.Embarked,train_df.Pclass],[train_df.Sex,train_df.Survived],margins=True).style.background_gradient(cmap='summer_r')
pd.crosstab([train_df.Sex,train_df.Survived],train_df.Pclass,margins=True).style.background_gradient(cmap='summer_r')
def age_buckets(x): 

    if x < 18: return '0-18' 

    elif x < 30: return '18-29'

    elif x < 40: return '30-39' 

    elif x < 50: return '40-49' 

    elif x < 60: return '50-59' 

    elif x < 70: return '60-69' 

    elif x >=70: return '70+' 

    else: return 'other'

    

    

train_df["Age"] = train_df["Age"].astype(int)



train_df["Age_Range"] = train_df["Age"].apply(lambda x: age_buckets(x))



fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,8))  



grouped_by_age_female = train_df[train_df["Sex"] == 'female'].groupby(["Age_Range"])["Survived"].value_counts().unstack()

grouped_by_age_female.plot.bar(stacked=True, color=['#99CCFF', '#BCE2C8'], rot=0,ax=ax1, title="Number of female survived/drowned passengers age group")

ax1.legend(('Drowned', 'Survived'))



grouped_by_age_men = train_df[train_df["Sex"] == 'male'].groupby(["Age_Range"])["Survived"].value_counts().unstack()

grouped_by_age_men.plot.bar(stacked=True, color=['#99CCFF', '#BCE2C8'], rot=0,ax=ax2, title="Number of male survived/drowned passengers per age group")

ax2.legend(('Drowned', 'Survived'))
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,8)) 



grouped_by_family = train_df.groupby(["Parch"])["Survived"].value_counts()

grouped_by_family.unstack().plot.bar(stacked=True, color=['#99CCFF', '#BCE2C8'], rot=0,ax=ax1, title="Number of survived/drowned passengers per number of parents/children on board")

plt.legend(( 'Drowned', 'Survived'), loc=(1.04,0))

ax1.legend(('Drowned', 'Survived'))



grouped_by_family_norm = train_df.groupby(["Parch"])["Survived"].value_counts(normalize=True)

grouped_by_family_norm.unstack().plot.bar(stacked=True, color=['#99CCFF', '#BCE2C8'], rot=0,ax=ax2, title="Proportion of survived/drowned passengers per number of parents/children on board")

ax2.legend(('Drowned', 'Survived'))
grouped_by_sibsp = train_df.groupby('SibSp')['Survived'].value_counts(normalize=True).unstack()

grouped_by_sibsp.plot(kind='bar', color=["#99CCFF", "#BCE2C8"], stacked=True, rot=0, figsize=(10,8), title="Number of survived/drowned passengers per number of siblings/spouses on board")

plt.legend(( 'Drowned', 'Survived'))



plt.xlabel('Number of siblings/spouses')

plt.ylabel('%')

plt.show()
#Sex and Embarked columns to check if there's any correlation between them and the Survived data



train_df['Sex_data'] = train_df['Sex'].map({'male': 1,'female': 0})

train_df['Embarked_data'] = train_df['Embarked'].map({'S': 0,'C': 1, 'Q': 2})

train_df.corr()
sns.heatmap(train_df.corr(), cmap='icefire')

plt.title('Correlation', fontsize=24)

#In our analysis we see a low correlation between embarked and sex.
#Visualization of 'Survived' (Target column)

train_df.Survived.value_counts()
train = train_df.Survived.value_counts().plot(kind='bar')

train.set_xlabel('Survived or not')

train.set_ylabel('Passenger Count')

train = train_df[['Pclass', 'Survived']].groupby('Pclass').mean().Survived.plot(kind='bar')

train.set_xlabel('Pclass')

train.set_ylabel('Survival Probability')
#Survival per Age/Sex

grouped_by_sex = train_df.groupby(["Sex"])["Survived"].value_counts()

grouped_by_sex.unstack().plot.bar(stacked=True, color=['#99CCFF', '#BCE2C8'], rot=0)