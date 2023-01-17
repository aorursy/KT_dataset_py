df.head()
df.info()
df.rename(columns={'NEmbarked':'Embarked'}, inplace=True)
df.info()
df.drop('Embarked', axis=1, inplace=True)
df.head()
df.info()
df['NEmbarked']=df['Embarked'].map({'S':1,'C':2,'Q':3})
df['Embarked'].value_counts()
print(df['Age'].value_counts().sum())

df.info()



import pandas as pd

import numpy as np



dftrain=pd.read_csv("../input/train.csv")

dftest=pd.read_csv("../input/test.csv")

dftrain.head()

dftest.head()

df=pd.concat([dftrain,dftest])

df.head()

df.info()

df.reset_index(drop=True,inplace=True)

df['Age'].interpolate(inplace=True)

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







# Any results you write to the current directory are saved as output.