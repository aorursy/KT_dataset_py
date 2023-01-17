# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import numpy as np 

import seaborn as sns

import pandas as pd 

import matplotlib.pyplot as plt 

sns.set(style = 'whitegrid')
tips = sns.load_dataset('tips')

tips.head()
sns.relplot(kind = 'scatter',x = 'total_bill',y = 'tip' , data  = tips,color ='k');
sns.relplot(x='total_bill',y = 'tip',hue = 'smoker',data= tips,palette = ['k','gray'],height = 6);
sns.relplot (x = 'total_bill',y = 'tip',hue = 'smoker',style = 'smoker' ,palette = ['k','gray'],data = tips);
sns.relplot (x = 'total_bill',y = 'tip',hue = 'smoker',style = 'day' ,palette = ['k','gray'],data = tips);
sns.relplot (x = 'total_bill',y = 'tip',hue = 'size',style = 'day' ,palette = "copper",data = tips);
sns.relplot(x='total_bill',y='tip',data = tips ,size = 'size' ,color = 'k');
df = pd.DataFrame()

df['x'] = np.arange(500)

df['y'] = (df['x'].values)**2
sns.relplot(kind = 'line' ,x='x',y='y',data=df,color='k',height=6);
sns.relplot(x='x',y='y',data=df,color='k',height=6);
df1 = pd.DataFrame()

df1['x'] = np.random.randn(500)

df1['y'] = np.random.randn(500).cumsum()
sns.relplot(x='x',y='y',color='k',data = df1,kind='line',sort=False,height= 6);
sns.relplot(x='x',y='y',color='k',data = df1,kind='line',height= 6);
fmri = sns.load_dataset("fmri")

fmri.head()
fmri['subject'].unique()
sns.relplot(x='timepoint',y = 'signal' , data = fmri,color='k',height = 6);
sns.relplot(x='timepoint',y = 'signal' , data = fmri,kind = 'line',color = 'k',height=6);
sns.relplot(x='timepoint',y = 'signal' , data = fmri,kind = 'line',color = 'k',ci = 'sd',height=6);
sns.relplot(x='timepoint',y = 'signal' , data = fmri,kind = 'line',color = 'k',ci = None,estimator = None,height = 6);
sns.relplot(x="timepoint", y="signal", hue="event", kind="line", data=fmri,palette = ['k','gray']);
sns.relplot(x="timepoint", y="signal", hue="event", style="region",

            kind="line",dashes = True,data=fmri,height = 6,palette = 'bone');
df1['z'] = np.arange(500)
dots = sns.load_dataset("dots").query("align == 'dots'")
sns.relplot(x="time", y="firing_rate",

            size="coherence", style="choice",

            kind="line", data=dots,color = 'k',height = 6);
df = pd.DataFrame(dict(time=pd.date_range("2017-1-1", periods=300),

                       value=np.random.randn(300).cumsum()))

g = sns.relplot(x="time",kind='line', y="value", data=df,color = 'k',height = 6) 

g.fig.autofmt_xdate(); #g is passed to the matplotlib function to format to the data into dates format 
tips.head()
sns.relplot(x='total_bill',y='tip',col='sex',hue='size',data=tips,palette = 'bone');
sns.relplot(x='total_bill',y='tip',col='sex',hue='size',row='smoker',data=tips,height=3 ,palette = 'bone');