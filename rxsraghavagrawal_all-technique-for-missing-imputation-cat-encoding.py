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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
data = train.copy()
data.head(3)
data.dtypes
data.drop('Name',axis=1,inplace=True)
# first let's see the distribution of target variable

sns.distplot(data['Survived'])
data.isnull().sum()
# Imputing age with the best possible technique

sns.distplot(data['Age'])

plt.show()
median = data.Age.median()

median
def impute_median(variable):

    data[variable+'_median'] = data[variable].fillna(data[variable].median())

    

impute_median('Age')
data['Age_median'].isna().sum()
sns.distplot(data['Age'],hist=False,color='red')

sns.distplot(data['Age_median'],hist=False,color='green')

plt.show()
# i am going to read the data again with only some column and we will best find the imputation technique

df = pd.read_csv('/kaggle/input/titanic/train.csv',usecols=['Age','Fare','Survived'])
df.head()

# here i am going to perform the random sampling imputation
df.Age.sample()

# we can see that it takes the random sample from the Age everytime we run it
df['Age'].dropna().sample(df.Age.isna().sum(),random_state=1)
def impute_nan(df,variable,median):

    df[variable+'_median'] = df[variable].fillna(median)

    df[variable+'_random'] = df[variable]

    random_sample = data['Age'].dropna().sample(data[variable].isna().sum(),random_state=1)

    

    random_sample.index = df[df[variable].isnull()].index

    df.loc[df[variable].isnull(), variable + '_random'] = random_sample

median = df.Age.median()

impute_nan(df,'Age',median)
df.head()
sns.distplot(df['Age'],hist=False,color='red')

sns.distplot(df['Age_median'],hist=False,color='green')

sns.distplot(df['Age_random'],hist=False,color='blue')

plt.show()
df['Age_nan'] = np.where(df['Age'].isnull(),1,0)

# where the value is null place 1 otherwise 0
df.head()
# sns.distplot(df['Age'])

df['Age'+'Zeros'] = df['Age'].fillna(0)

df['Age'+'hund'] = df['Age'].fillna(100)
extreme = df['Age'].mean() + 3*df['Age'].std()

print(extreme)

df['Age'+'_end'] = df['Age'].fillna(extreme)
sns.distplot(df['Age'])
sns.distplot(df['Age_end'])

plt.show()
data = pd.read_csv('/kaggle/input/titanic/train.csv')
categorical_features= [feature for feature in data.columns if data[feature].dtype == 'O']

print("len of cat feat:",len(categorical_features))
data[categorical_features].head()
data[categorical_features].isna().sum()
data['Embarked'].value_counts()

# S has highest frequency
def impute_freq_cat(df,variable):

    freq_cat = df[variable].value_counts().index[0]

    df[variable].fillna(freq_cat,inplace=True)

    

impute_freq_cat(data,'Embarked')
data['Embarked'].isna().sum()

# Ok it's been imputed correctly
data['cabin_nan'] = np.where(data['Cabin'].isna(),1,0)

data.head()
data['Cabin'] = data['Cabin'].fillna('Missing')

data.head()
data.head()
data[categorical_features].head()
pd.get_dummies(data['Sex'])
# here the one column is suitable to represent the other column as well so there is no need og it

sex = pd.get_dummies(data['Sex'],drop_first=True)

#sex
data['Embarked'].unique()
def create_cat_mapping(df,variable):

    return {k: i for i, k in enumerate(df[variable].unique(),0)}
mapping = create_cat_mapping(data,'Embarked')

mapping

# nice it creates the dictonary in which we can mapp it easily
data['embarked_impute'] = data['Embarked'].map(mapping)
data['Embarked'].value_counts(normalize=True)
counts = data['Embarked'].value_counts().to_dict()

data['Embarked_valcount'] = data['Embarked'].map(counts)
df = pd.read_csv('/kaggle/input/titanic/train.csv',usecols=['Cabin','Survived'])

df.head()
df['Cabin'].fillna('Missing',inplace=True)
df['Cabin'] = df['Cabin'].astype(str).str[0]
df.head(3)
df['Cabin'].unique()
df.groupby('Cabin')['Survived'].mean()
order_labels = df.groupby('Cabin')['Survived'].mean().sort_values().index

order_labels
mapping_dict = {k: i for i, k in enumerate(order_labels,0)}

mapping_dict
df['Cabin_impute'] = df['Cabin'].map(mapping_dict)

df.head()
mean_ordinal = df.groupby('Cabin')['Survived'].mean().to_dict()

mean_ordinal
df['Cabin_mean'] = df['Cabin'].map(mean_ordinal)

df.head()
df = pd.read_csv('/kaggle/input/titanic/train.csv',usecols=['Cabin','Survived'])

df.head()
df['Cabin'].fillna('Missing',inplace=True)

df['Cabin'] = df['Cabin'].astype(str).str[0]
prob_df = df.groupby('Cabin')['Survived'].mean()

prob_df
prob_df = pd.DataFrame(prob_df)

prob_df
prob_df['Not-Survived'] = 1 - prob_df['Survived']

prob_df
prob_df['prob_ratio'] = prob_df['Survived'] / prob_df['Not-Survived']

prob_df.head()
prob_encoded = prob_df['prob_ratio'].to_dict()

#prob_encoded
df['Cabin_encoded'] = df['Cabin'].map(prob_encoded)

df.head()
# i am performing directly the steps because in the above line we have calculated the probability ratio

df['cabin_woe'] = np.log(df['Cabin_encoded'])

df.head()