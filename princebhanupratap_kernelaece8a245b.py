import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sb
data=pd.read_csv('../input/StudentsPerformance.csv')
data.head()
df=data['race/ethnicity'].value_counts()

df
data.shape
df.plot(kind='bar')
pr=data.groupby('parental level of education').mean()

pr
pr.plot(kind='bar' , figsize=(15,7))
test=data.groupby('test preparation course').mean()

test
test.plot(kind='barh')
test=data.groupby('gender').mean()

test
test.mean(axis=1).plot(kind='bar')
data.groupby(['gender','race/ethnicity']).mean()
data.groupby(['parental level of education','gender']).mean()
sb.barplot(x='race/ethnicity' , y='math score',data=data)