import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

print(train_df.head())

print(test_df.head())
#--- Memory usage of entire dataframe ---

mem = train_df.memory_usage(index=True).sum()

print(mem/ 1024," KB")
#--- List of columns that CAN be reduced in terms of memory size ---

count = 0

red_col = []

for col in train_df.columns:

    if train_df[col].dtype != object:

        count+=1

        red_col.append(col)

        

print('There are {} columns that can be reduced'.format(count))   

print (red_col)
#--- Identifying the datatypes of each of these columns ---

for i in red_col:

    print(i + ' - {}'.format(train_df[i].dtype))
print(train_df['Fare'].describe())

print(train_df['Age'].describe())
count = 0

for col in red_col:

    if train_df[col].dtype == np.float64:

        count+=1

        train_df[col] = train_df[col].astype(np.float32)

        print (col)

print(count)                
#--- Let us check how much memory we have reduced in consumption ---

mem = train_df.memory_usage(index=True).sum()

print(mem/ 1024," KB")
print(train_df['PassengerId'].describe())

print(train_df['Pclass'].describe())

print(train_df['Survived'].describe())

print(train_df['Parch'].describe())

print(train_df['SibSp'].describe())

#--- Convert to type int8 ---

train_df['Parch'] = train_df['Parch'].astype(np.int8)

train_df['Survived'] = train_df['Survived'].astype(np.int8)

train_df['Pclass'] = train_df['Pclass'].astype(np.int8)



#--- Convert to type int16 ---

train_df['PassengerId'] = train_df['PassengerId'].astype(np.int16)

train_df['SibSp'] = train_df['SibSp'].astype(np.int16)
#--- Let us again check how much memory we have reduced in consumption ---

mem = train_df.memory_usage(index=True).sum()

print(mem/ 1024," KB")
print(train_df['Sex'].unique())

print(train_df['Embarked'].unique())
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)



#--- Checking if null values still exist ---

train_df['Embarked'].isnull().sum()
#--- Replacing the unique string  values with numerical values ---



train_df['Sex'].replace( 'male', 1, inplace=True)

train_df['Sex'].replace( 'female', 0, inplace=True)



#--- Convert the column to type `int8` ---

train_df.Sex = train_df.Sex.astype(np.int8)



train_df['Embarked'].replace( 'S', 1, inplace=True)

train_df['Embarked'].replace( 'C', 2, inplace=True)

train_df['Embarked'].replace( 'Q', 3, inplace=True)



#--- Convert the column to type `int8` ---

train_df.Embarked = train_df.Embarked.astype(np.int8)
print(train_df.dtypes)
#--- Let us again check how much memory we have reduced in consumption ---

mem = train_df.memory_usage(index=True).sum()

print(mem/ 1024," KB")

train_df.to_csv('Titanic_reduced_train_set.csv')

print('DONE!!')