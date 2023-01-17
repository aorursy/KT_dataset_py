# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#understanding pictureof the data

df=pd.read_csv("../input/master.csv")

df.head()

print("the number of rows are: "+str(df.shape[0]))

print("the number of column are: "+str(df.shape[1]))



#finding unique countries entry

unique_country=df['country'].unique()

print(unique_country)

print("the number of unique countries in the dataset are: "+str(len(unique_country)))
#finding correlation between data

correlation_between_attributesofData=sns.heatmap(df.corr(),annot=True)
plt.figure(figsize=(10,5))

p = sns.barplot(x='age', y='suicides/100k pop', hue='sex', data=df)
g=sns.lineplot(x='year',y='suicides/100k pop',hue='sex',data=df.groupby(['year','sex']).sum().reset_index()).set_title('graph')
p=sns.barplot(x='sex',y='suicides/100k pop',hue='age',data=df)