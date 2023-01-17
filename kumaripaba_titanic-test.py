# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

type(df)

df.head()
df.shape

df.shape[0]

df.shape[1]

df.columns
df.describe()
df['Age'].describe()

df['Age'][df['Age']>10] 

children = df['Age'] <= 10

children.sum()
df['Age'].plot.hist()
df['Fare'].plot.hist(bins=50)
embarked = df['Embarked']

embarked.value_counts()
class_count = df['Pclass'].value_counts()

class_count.plot.pie()
