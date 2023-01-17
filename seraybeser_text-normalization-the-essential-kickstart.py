import numpy as np 

import pandas as pd 
df = pd.read_csv("../input/en_train.csv")
df.head(10)
df['class'].unique()
tdf = pd.read_csv("../input/en_test.csv")

tdf.head(20)
import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

f,axarray = plt.subplots(2,1,figsize=(15,10))

hist = df.groupby('class',as_index=False).count()

hist = hist[hist['class']!='PLAIN']

g= sns.barplot(x=hist['class'],y=hist['before'],ax=axarray[0])

for item in g.get_xticklabels():

    item.set_rotation(45)

hist = hist[hist['class']!='PUNCT']

g= sns.barplot(x=hist['class'],y=hist['before'],ax=axarray[1])

for item in g.get_xticklabels():

    item.set_rotation(45)

plt.show()
length = df.groupby(['sentence_id'],as_index=False).count()

length = length.groupby(['before'],as_index=False).count()
length['before'].describe()
f,axarray = plt.subplots(1,1,figsize=(15,10))

length = length[0:40]

sns.barplot(x = length['before'],y=length['after'])

df[df['class']=='PUNCT'].head()
len(df[df['class']=='DATE'])
df[df['class']=='DATE'].head(10)
len(df[df['class']=='PUNCT'])
df[df['class']=='LETTERS'].head()
len(df[df['class']=='LETTERS'])
df[df['class']=='CARDINAL'].head()
len(df[df['class']=='CARDINAL'])
df[df['class']=='VERBATIM'].head()
len(df[df['class']=='VERBATIM'])
df[df['class']=='DECIMAL'].head()
len(df[df['class']=='DECIMAL'])
df[df['class']=='MEASURE'].head()
len(df[df['class']=='MEASURE'])
df[df['class']=='MONEY'].head()
len(df[df['class']=='MONEY'])
df[df['class']=='ORDINAL'].head()
len(df[df['class']=='ORDINAL'])
df[df['class']=='TIME'].head()
len(df[df['class']=='TIME'])
df[df['class']=='ELECTRONIC'].head()
len(df[df['class']=='ELECTRONIC'])
df[df['class']=='DIGIT'].head()
df[df['class']=='FRACTION'].head()
len(df[df['class']=='FRACTION'])
df[df['class']=='TELEPHONE'].head()
len(df[df['class']=='TELEPHONE'])
df[df['class']=='ADDRESS'].head()
len(df[df['class']=='ADDRESS'])