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
# Import our libraries we are going to use for our data analysis.

import tensorflow as tf

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt



# Plotly visualizations

from plotly import tools

import plotly.plotly as py

import plotly.figure_factory as ff

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

# plotly.tools.set_credentials_file(username='AlexanderBach', api_key='o4fx6i1MtEIJQxfWYvU1')





# For oversampling Library (Dealing with Imbalanced Datasets)

from imblearn.over_sampling import SMOTE

from collections import Counter



# Other Libraries

import time
df= pd.read_csv('../input/lc_2016_2017.csv', low_memory=False)

df.head()
df_sample = df.sample(frac=0.1)

df_sample.shape
df_sample.isnull().sum()
df_sample1=df_sample.dropna(thresh=df_sample.shape[0]*0.9,how='all',axis=1)

df_sample1.isnull().sum()
df_sample1.shape
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

from collections import Counter


trace0 = go.Bar(

    x = df_sample1.emp_title.value_counts()[:30].index.values,

    y = df_sample1.emp_title.value_counts()[:30].values,

    marker=dict(

        color=df_sample1.emp_title.value_counts()[:30].values

    ),

)



data = [trace0]



layout = go.Layout(

    yaxis=dict(

        title='Count'

    ),

    xaxis=dict(

        title='Employment name'

    ),

    title='TOP 40 Employment Title'

)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='emp-title-bar')
print(pd.crosstab(df_sample1["emp_length"], df_sample1["application_type"]))



fig, ax = plt.subplots(2,1, figsize=(12,10))

g = sns.boxplot(x="emp_length", y="int_rate", data=df_sample1,

              palette="hls",ax=ax[0],

               order=["n/a",'< 1 year','1 year','2 years','3 years','4 years', '5 years',

                      '6 years', '7 years', '8 years','9 years','10+ years'])



z = sns.violinplot(x="emp_length", y="loan_amnt",data=df_sample1, 

               palette="hls", ax=ax[1],

               order=["n/a",'< 1 year','1 year','2 years','3 years','4 years', '5 years',

                      '6 years', '7 years', '8 years','9 years','10+ years'])

               

plt.legend(loc='upper left')

plt.show()
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

plt.subplots_adjust(wspace=1.0, hspace=0.50)

df_sample1.emp_length.value_counts().plot(kind="bar", ax=ax[0])

ax[0].set_title("Employment Length Count")

df_sample1.purpose.value_counts().plot(kind="bar", ax=ax[1])

ax[1].set_title("Loan Purposes")

plt.xticks(rotation=60)
df_sample1.emp_title.value_counts()

df_sample1.title.value_counts()

df_sample2 = df_sample1.dropna(subset=['revol_util','dti', 'title','mths_since_rcnt_il','all_util','inq_fi','total_cu_tl','inq_last_12m'])

df_sample2.isnull().sum()

df_sample3=df_sample2.drop(['last_pymnt_d','emp_title'],axis=1)
df_sample3["emp_length"]=df_sample3["emp_length"].fillna("< 1 year")
tot_cel = df_sample3.isnull().sum().sum()

print(tot_cel)

plt.figure(figsize=(15,4))

sns.heatmap(df_sample3.isnull(), cbar = False, yticklabels=False, cmap="magma" )
df_sample3.isnull().sum()