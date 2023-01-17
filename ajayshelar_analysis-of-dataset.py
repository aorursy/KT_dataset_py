# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
pd.read_csv('../input/train.csv').head()
df = pd.read_csv('../input/train.csv')
df.describe()
df.head()
df[df['Sex'] == 'female']
len(df[df['Sex'] == 'female'])

len(df[df['Sex'] == 'male'])
print(len(df[(df['Sex'] == 'female') & (df['Age'] > 18)]))
df[(df['Sex'] == 'female') & (df['Age'] > 18) ]
print(len(df[(df['Sex'] == 'female') & (df['Age'] <18)]))
df[(df['Sex'] == 'female') & (df['Age'] > 18) ]

print(len(df[(df['Sex'] == 'female') & (df['Fare']>50 ) &(df['Age'] > 18) & (df['Survived'] == 1)]))
df[(df['Sex'] == 'female') & (df['Fare']>50 ) & (df['Survived'] == 1)]
# df[df['Fare']>50]
print(len(df[(df['Sex'] == 'female') & (df['Fare']<50 ) & (df['Survived'] == 1)]))
df[(df['Sex'] == 'female') & (df['Fare']<50 ) & (df['Survived'] == 1)]
df[pd.notnull(df['Cabin'])]
