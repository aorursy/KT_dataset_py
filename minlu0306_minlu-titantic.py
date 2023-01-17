import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

train.head()
train.info()

print ("-----")

test.info()
train_df=train.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)

train_df
train_df.describe()
#now let's handle some missing values in data:

train_df.isnull().sum()
train_df['Embarked'].describe()
train_df['Embarked']=train_df['Embarked'].fillna('S')
#There are 177 missing ages in this data. we will use random values that are within 1

#stardard deviation of the mean to fill these values manually.

mean = train_df['Age'].mean()

std = train_df['Age'].std()



replace_nan = lambda x: np.random.randint(mean-std, mean+std) if np.isnan(x) else x



train_df['Age2'] = train_df['Age'].apply(replace_nan)
train_df
train_df.drop('Age',axis=1)
def adjust_sex_to_numbers(n):

    if n=='male':

        n=0

    elif n=='female':

        n=1

    else:

        n=n

    return n
train_df['Sex'].apply(adjust_sex_to_numbers)
#now, let start calculate correlation for each variables:

def correlation(x,y):

    a=(x-x.mean())/x.std(ddof=0)

    b=(y-y.mean())/y.std(ddof=0)

    return (a*b).mean()
sex=train_df['Sex'].apply(adjust_sex_to_numbers)

survive=train_df['Survived']

class1=train_df['Pclass']

age=train_df['Age']

sibsp=train_df['SibSp']

parch=train_df['Parch']

fare=train_df['Fare']
correlation(survive,sex)
correlation(survive,age)
correlation(survive,sibsp)
correlation(survive,parch)
correlation(survive,fare)