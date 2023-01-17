from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR6ai8THI0siaak5Xa_n9E90Etq1oiA5Ah9IlArHGLvwmdPbSXK&usqp=CAU',width=400,height=400)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML

import plotly.express as px

import plotly.graph_objects as go

import seaborn as sns

from plotly.subplots import make_subplots

%matplotlib inline

import missingno as msno 



import plotly.tools as tls

import cufflinks as cf

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True)



print(__version__) # requires version >= 1.9.0

cf.go_offline()



df = pd.read_csv('../input/uncover/UNCOVER/us_cdc/us_cdc/nutrition-physical-activity-and-obesity-behavioral-risk-factor-surveillance-system.csv', encoding='ISO-8859-2')

df.head()
df.plot.area(y=['yearstart','yearend','locationid'],alpha=0.4,figsize=(12, 6));
df.corr()

plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=True,cmap='YlOrRd_r')

plt.show()
msno.bar(df) 
msno.heatmap(df)
msno.matrix(df)
plt.clf()

df.groupby('class').size().plot(kind='bar')

plt.show()
plt.clf()

df.groupby('class').sum().plot(kind='bar')

plt.show()
plt.clf()

df.groupby('stratificationcategory1').size().plot(kind='bar')

plt.show()
df_st_ct = pd.value_counts(df['locationdesc'])

df_st_ct 
df.describe().T
df.info()
df.corr()
df.isnull().sum()
sns.heatmap(df[df.columns[:]].corr(),annot=True,cmap='RdYlGn')

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
import itertools

columns=df.columns[:8]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    df[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
df.drop('data_value_unit', axis=1, inplace=True)

sns.pairplot(data=df,hue='class',diag_kind='kde')

plt.show()