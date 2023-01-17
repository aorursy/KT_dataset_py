import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv("/kaggle/input/top250transfers/top250-00--19.csv")#reading the csv file from our file location
df.head()#seeing the datas present in our data frame
df.shape  #checking number of rows and columns in a data
df.describe()#description of the dataframe
max_fee=df.Transfer_fee.max()

filt=df["Transfer_fee"]==max_fee

df[filt]
import seaborn as sns

df.loc[(df.Season>="2000")].groupby("Team_to").Transfer_fee.agg("sum").sort_values(ascending=False).head(5).plot.bar(alpha=0.75)
df.hist()

plt.tight_layout()
sns.countplot(x='Position',data=df)

plt.xticks(rotation=90)
sns.heatmap(df.corr(),annot=True)
# From the above heatmap we can see that market_value of a player affected the transfer fee most.
#Conclusion-->

#this is my first ever project . I have just started learning Ml & Data_Sc . Just a little effort to start of my journey.

#Most of the codes are inspired from other people's works.