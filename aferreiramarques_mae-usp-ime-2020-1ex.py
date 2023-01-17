import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv(r"325.csv", sep=';')
pd.set_option('display.max_columns', None)  
df.head(6)
df.info()
! pip install plotly_express
import plotly_express as px
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (20, 8),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

%matplotlib inline
plt.rcParams.update(params)
fig=df.plot('Lamp', 'Quant', kind='bar')
df.describe
print(df['Lamp'].mean())
print(df['Lamp'].mode())
print(df['Lamp'].max() - df['Lamp'].min())
print(df['Lamp'].median())
print(round(df['Lamp'].var(),3))
print(round(df['Lamp'].std(),3))
print(df['Lamp'].quantile(q=0.1))
print(df['Lamp'].quantile(q=0.3))
print(df['Lamp'].quantile(q=0.5))
print(df['Lamp'].quantile(q=0.7))
print(df['Lamp'].quantile(q=0.9))
c=1
while c<9.5:
    print(df['Lamp'].quantile(q=c/10))
    c=c+1
def quantile(n):
    A=print(df['Lamp'].quantile(q=n/10))
    return A
b=quantile(3)
print(b)
c=quantile(6)
print(c)
