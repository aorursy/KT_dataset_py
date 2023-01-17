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
## Importing the datasets

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
#get the shape of the dataframe 

train.shape
#get the columns of the data frame

train.columns
#get the data type of the columns using dtypes attribute

train.dtypes
train.info()
train.describe()
print(f"train is of type : {type(train)} ")

print(f"shape attribute returns : {type(train.shape)} ")

print(f"columns attribute returns : {type(train.columns)} ")

print(f"dtypes attribute returns : {type(train.dtypes)} ")



#head method by default gives the first/top five rows of the dataframe

train.head()  # index 0 to 4
#the number of rows returned from head method can be controlled by passing the argument n

train.head(n=4)
train.head(3)
#tail method is similar to head but fetches rows from the bottom



train.tail()
train.tail(n=3)
train.tail(4)
# sample() randomly selects the rows and displays and the default number of rows displayed is 1



train.sample()
train.sample(n=4)
train.sample(2)
# display first 10 rows

train[0:10] # 0 inclusive and 10 exclusive
# just display name column

train['Name'].sample(5)
# display multiple columns

cols = ['Name','Sex','Age']

train[cols].sample(5)
#loc for displaying a particular row



#display 99th row



train.loc[98]  #index starts from 0
#display 1st,25th,50th rows



train.loc[[0,24,49]]
#display from 15th row to 20th

# In loc the start and end are inclusive

train.loc[14:19]
#display from 15th row to 20th only Name

train.loc[14:19]['Name']
#display from 15th row to 20th only Name, Age

train.loc[14:19][['Name','Age']]
#display from 15th row to 20th and all columns between name and embarked

train.loc[14:19,'Name':'Embarked']
# to understand the loc better let us replace the default index and make the name as index just for demonstration purpose

train_temp = train.set_index('Name',inplace=False)
# can see that now the row index is Name and is no more the default index 

train_temp.head()
#loc using the new name index

train_temp.loc['Braund, Mr. Owen Harris']
# display rows using the range

train_temp.loc['Braund, Mr. Owen Harris':'Allen, Mr. William Henry']
#displaying rows using loc range and filter columns as well

train_temp.loc['Braund, Mr. Owen Harris':'Allen, Mr. William Henry', 'PassengerId':'Embarked']
# select 0 to 9 rows with all the columns



train.iloc[0:10] # 0 is inclusive and 10 is exclusive
# select 0 to 9 rows with all the columns



train.iloc[0:10,:]
# select 0 to 9 rows 3 to 5 columns



train.iloc[0:10,3:6]  # 0 inclusive,10 exclusive, 3 inclusive, 6 exclusive
# select particular row

train.iloc[10,:]
train_temp.loc['Allen, Mr. William Henry','Age']
train_temp.loc[train_temp.Age==25]
train_temp.loc[(train_temp.Age==25) & (train_temp.Sex=='male')]
train.iloc[-100:, -8:]
grouped_by_age = train.groupby('Sex')

grouped_by_age.sample(5)
train.groupby('Sex')['Age'].mean()
train[train.Survived==1].groupby('Sex').nunique()
print(train[(train.Survived==1) & (train.Age>=50)].shape[0])

print(len(train[(train.Survived==1) & (train.Age>=50)]))
train.groupby('Sex')['Age'].value_counts()
train.sort_values('Name').head(5)
train.sort_values('Name').tail(5)
train.rename(columns={'Sex':'Gender','Fare':'Cost'}, inplace=False)  #inplace is False and hence this change is temporary
pd.crosstab(train['Sex'],train['Survived'])
pd.crosstab(train['Survived'],train['Sex'])
# get the mean age of the survived

train[train.Survived ==1]['Age'].mean()
# get the mean age of all the male survived

train[(train.Survived ==1) & (train.Sex=='male')]['Age'].mean()
# get the mean age of all the fe-male survived

train[(train.Survived ==1) & (train.Sex=='female')]['Age'].mean()
#get all the rows where embarked value = S

train_temp = train.dropna()

train_temp[train_temp.Embarked=='S']
train.loc[10:20:,'Name':'Embarked']
train.iloc[10:21:,3:12]
train.groupby('Sex')['Fare'].sum()
train.groupby('Survived')['Fare'].sum()
train['Sex'].value_counts()
train.groupby('Embarked')['Fare'].mean()
train_survived = train[train['Survived']==1]

train_survived['Sex'].value_counts()
def categorise_age(age):

    if age <= 12 : 

        return 'child'

    if age <= 20 : 

        return 'teen'

    if age >50 :

        return 'senior'

    else :

        return 'adult'



train['age_group']=train.Age.apply(lambda x :  categorise_age(x))
train.age_group.value_counts()
import matplotlib as plt

%matplotlib inline

import seaborn as sns
sns.countplot(train.Sex)
sns.countplot(train.age_group)
sns.countplot(train.Survived)
sns.catplot(x='Sex',y='Survived',kind='bar',data=train)
sns.catplot(x='age_group',y='Survived',kind='bar',data=train)
corr_matrix = train.corr()

corr_matrix['Survived']
sns.heatmap(corr_matrix)