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
import pandas as pd

import numpy as nm

from matplotlib import pyplot as plt
df=pd.read_csv("../input/StudentsPerformance.csv")
df.head()
import seaborn as sb
data=df.groupby('race/ethnicity').mean()

data
a=df['gender'].value_counts()

a
a.plot(kind='bar')
data.mean(axis=1).plot(kind='bar')
a=df.groupby('parental level of education').mean()

a
a.plot(kind='bar')
a.mean(axis=1).plot(kind='bar')
a=df.groupby('test preparation course').mean()

a
a.plot(kind='bar')
a=df.groupby('gender').mean()

a
#average performance of both the genders

a.mean(axis=1).plot(kind='bar')
a=df.groupby(['parental level of education','gender']).mean()

a
a.mean(axis=1).plot(kind='bar')
a=df.groupby(['gender','race/ethnicity']).mean()

a
sb.barplot(x='race/ethnicity', y='math score', data=df)
sb.barplot(x='race/ethnicity', y='reading score', data=df)
sb.barplot(x='race/ethnicity', y='reading score', data=df)
a= df.drop(['lunch','test preparation course'], axis=1)

a.head()