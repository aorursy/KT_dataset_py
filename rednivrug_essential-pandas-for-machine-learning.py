import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

print(os.listdir("../input/"))
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
pd.Series([5,6,12,-5,6.7],index=['A','B','C','D','E'],dtype=np.float)
a=[1000,1001,1002,1003,1004]

b=['Steve','Mathew','Jose','Patty','Vin']

c=[86.29,91.63,72.90,69.23,88.30]

pd.DataFrame(list(zip(a,b,c)),columns=['Regd. No','Name','Marks%'])
df =  pd.read_csv("../input/Pokemon.csv")
print('The columns of the dataset are: ',df.columns)

print('The shape of the dataframe is: ',df.shape)
df.head(3)
df.dtypes
df[["Name","Type 1"]].head(3)
df.describe()
df.describe(include=['object'])
df = df.set_index('Name')

df.head(3)
df=df.drop(['#'],axis=1)

df.head(3)
df[df.index.str.contains("Mega")].head(3)
df.columns = df.columns.str.upper().str.replace('_', '')

df.head(3)
corr = df.corr()

corr.style.background_gradient()
df[df['LEGENDARY']==True].head(3)
df['TYPE 2'].fillna(df['TYPE 1'], inplace=True)
df.loc['Bulbasaur']
df.iloc[0]
df.ix[0]
df.ix['Kakuna']
df[((df['TYPE 1']=='Fire') | (df['TYPE 1']=='Dragon')) & ((df['TYPE 2']=='Dragon') | (df['TYPE 2']=='Fire'))].head(3)
print("MAx HP:",df['HP'].idxmax())

print("Max DEFENCE:",(df['DEFENSE']).idxmax())
df.sort_values('TOTAL',ascending=False).head(3)
print('The unique  pokemon types are',df['TYPE 1'].unique())

print('The number of unique types are',df['TYPE 1'].nunique())
print(df['TYPE 1'].value_counts())
df.groupby(['TYPE 1']).size()
(df['TYPE 1']=='Bug').sum()
df[df["LEGENDARY"] == True].head(3)
df.loc[df["LEGENDARY"] == True,"SPEED"].head(3)
df.loc[df["LEGENDARY"] == True,"SPEED"] += 10

df.loc[df["LEGENDARY"] == True,"SPEED"].head(3)
df1 = pd.DataFrame({"A": ['A0','A1','A2','A3'],"B": ['B0','B1','B2','B3'],

        "C": ['C0','C1','C2','C3'],"D": ['D0','D1','D2','D3']})

s2 = pd.DataFrame({"A": ['X0'],"B": ['X1'], "C": ['X2'],"D": ['X3']})

df1

s2.T

Result = pd.concat([df1,s2],axis=0).reset_index()

Result.drop(['index'],axis=1)
pd.Timestamp('9/1/2016 10:05AM')
pd.Timestamp('9/3/2016')-pd.Timestamp('9/1/2016')
d1 = ['2 June 2013', 'Aug 29, 2014', '2015-06-26', '7/12/16']

pd.DataFrame(np.random.randint(10, 100, (4,2)), index=d1, columns=list('ab'))
f2 = pd.to_datetime(pd.Series(d1).values[0])

f2

f2.day

f2.month

f2.year
df["GENERATION"].value_counts().sort_index().plot.bar()
df["TYPE 1"].value_counts().plot.barh()
df.hist(column='ATTACK')
df['TYPE 1'].value_counts().plot.pie()