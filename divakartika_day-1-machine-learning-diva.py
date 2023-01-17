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
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df.head(5)
sum(df.isnull().sum() > 0)
total_male = (df['Sex'] == 'male').sum()

total_female = (df['Sex'] == 'female').sum()

percent_male = total_male / len(df)

percent_female = total_female / len(df)

print('Male Passanger Ratio is ',percent_male*100, '%')

print('Female Passanger Ratio is ',percent_female*100, '%')
df['Cabin'].nunique()
(df['Parch'] >= 2).sum()
no_sibling = df[df['SibSp'] > 0]

no_family = no_sibling[no_sibling['Parch'] > 0]

no_family['Pclass'].mode()
avg_fare = df.groupby('Embarked')['Fare'].mean().reset_index().rename(columns={'Fare':'AVG_Fare'})

avg_fare
old = df[df['Age'] >= 50]

old['Cabin'].mode()
df.groupby(['Pclass', 'Sex'])['Survived'].sum().to_frame(name='Survived')