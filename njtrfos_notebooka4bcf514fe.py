import sys
#!{sys.executable} -m pip install seaborn==0.9.0
import seaborn
import random
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#sns.set(rc={"figure.figsize": (24, 12)})
plt.style.use({'figure.figsize':(24, 12)})
sns.set_style('dark')
sns.set_context("poster")
#np.set_printoptions(suppress=True,   precision=10,  threshold=2000,  linewidth=150)  
pd.set_option('display.float_format',lambda x : '%.2f' % x)
plt.rcParams['axes.unicode_minus'] = False
from warnings import filterwarnings
filterwarnings('ignore')
np.set_printoptions(suppress=True,   precision=10)
file_name = '/kaggle/input/covid19-in-india/covid_19_india.csv'
df = pd.read_csv(file_name)
df['State'] = df['State/UnionTerritory']
df
df1 = df.groupby(['Date'])['Confirmed'].sum()
df1 = df1.to_frame().reset_index()
df1.sort_values('Date',ascending=True,inplace=True)
df1
df1.info()
plt.ticklabel_format(style='plain',axis='both')    
sns.lineplot(x=df1.index, y=df1.Confirmed,data=df1)
pt = df.pivot_table(index='Date', columns='State',  values=['Confirmed'],aggfunc=np.mean)
pt.fillna(value=0,inplace=True)
pt.head(10)
plt.style.use({'figure.figsize':(32, 12)})
plt.ticklabel_format(style='plain',axis='both')
cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(pt, cmap = cmap, linewidths = 0.05,annot=False, fmt="g")
df2 = df.groupby(['Date','State'])['Confirmed'].sum()
df2 = df2.to_frame().reset_index()
df2.sort_values('Date',ascending=True,inplace=True)
df2
ax = sns.lineplot(x=df2.index, y=df2.Confirmed, hue=df2.State, markers=True, data=df2)
ax.tick_params(axis='x', colors='b') # x轴
ax.set_xticklabels(df2['Date'], rotation=60)
ax.set_yticklabels(ax.get_xticklabels(),fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(),fontsize=16)
plt.grid(linestyle=':')
plt.show()