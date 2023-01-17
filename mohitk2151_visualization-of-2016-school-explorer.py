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
import matplotlib.pyplot as plt
import matplotlib
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import seaborn as sns 
import numpy as np
import pandas as pd
import numpy as np
import random as rnd
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score , average_precision_score
from sklearn.metrics import precision_score, precision_recall_curve
plt.style.use('fivethirtyeight')
%matplotlib inline
School_df = pd.read_csv("../input/2016 School Explorer.csv")
layout = go.Layout(xaxis=dict(title='City Name',tickangle=-35),
      yaxis=dict(title='Number of Schools'),title='Maximum Number of Schools-City Wise', width=1000, height=500, margin=dict(l=100))
trace1 = go.Bar(x=School_df['City'].value_counts().index, y=School_df['City'].value_counts().values, marker=dict(color="#FF7441"))

data = [trace1]
fig = go.Figure(data=data, layout=layout)
iplot(fig)

df = pd.DataFrame(School_df['Community School?'].value_counts().values,
                  index=School_df['Community School?'].value_counts().index, 
                  columns=[' '])

df.plot(kind='pie', subplots=True, autopct='%1.0f%%', figsize=(8, 8))
plt.title('Distribution of Schools-Community & Others')
plt.show()
School_df['School Income Estimate']=School_df['School Income Estimate'].replace({'\$':'', ',':''},regex=True).astype(float)
trace0 = go.Scatter(
    x=School_df[School_df['Community School?'] == 'Yes']['School Income Estimate'],
    y=School_df[School_df['Community School?'] == 'Yes']['Economic Need Index'],
    mode='markers',
    name='Community School? = Yes',
    marker=dict(
        size=2,
        line=dict(
            color='blue',
            width=10
        ),
        
    )
)
trace1 = go.Scatter(
    x=School_df[School_df['Community School?'] == 'No']['School Income Estimate'],
    y=School_df[School_df['Community School?'] == 'No']['Economic Need Index'],
    mode='markers',
    name='Community School? = No',
    marker=dict(
        size=2,
        line=dict(
            color='red',
            width=2.5
        ),
        
    )
)
data = [trace0, trace1]
layout = go.Layout(
      xaxis=dict(title='School Income Estimate'),
      yaxis=dict(title='Economic Need Index'),
      title=('Economic Need Assessment'))
fig = go.Figure(data=data, layout=layout)
iplot(fig)
t0 = go.Scatter3d(
    x=School_df[School_df['Community School?'] == 'Yes']['School Income Estimate'],
    y=School_df[School_df['Community School?'] == 'Yes']['Economic Need Index'],
    z=School_df[School_df['Community School?'] == 'Yes']['Grade High'],
    mode='markers',
    name='Community School? = Yes',
    marker=dict(
        size=2,
        line=dict(
            color='blue',
            width=10
        ),
        
    )
)
t1 = go.Scatter3d(
    x=School_df[School_df['Community School?'] == 'No']['School Income Estimate'],
    y=School_df[School_df['Community School?'] == 'No']['Economic Need Index'],
    z=School_df[School_df['Community School?'] == 'No']['Grade High'],
    mode='markers',
    name='Community School? = No',
    marker=dict(
        size=2,
        line=dict(
            color='red',
            width=2.5
        ),
        
    )
)
data = [trace0, trace1]
layout = go.Layout(
      xaxis=dict(title='School Income'),
      yaxis=dict(title='Economic Need Status'),
      title=('Economic Status-Community vs Private'))
fig = go.Figure(data=data, layout=layout)
iplot(fig)

School_df['Percent Black']=School_df['Percent Black'].replace({'\%':''},regex=True).astype(float)
School_df['Percent Black']=School_df['Percent Black']/100
School_df['Percent White']=School_df['Percent White'].replace({'\%':''},regex=True).astype(float)
School_df['Percent White']=School_df['Percent White']/100
School_df['Percent Asian']=School_df['Percent Asian'].replace({'\%':''},regex=True).astype(float)
School_df['Percent Asian']=School_df['Percent Asian']/100
School_df['Percent Hispanic']=School_df['Percent Hispanic'].replace({'\%':''},regex=True).astype(float)
School_df['Percent Hispanic']=School_df['Percent Hispanic']/100
School_df['Percent Black / Hispanic']=School_df['Percent Black / Hispanic'].replace({'\%':''},regex=True).astype(float)
School_df['Percent Black / Hispanic']=School_df['Percent Black / Hispanic']/100
no_comnt_school = School_df[School_df['Community School?'] == 'No']
comnt_school = School_df[School_df['Community School?'] == 'Yes']
v_features = ['Percent Hispanic','Percent Black','Percent White','Percent Asian']
plt.figure(figsize=(15,8))
gs = gridspec.GridSpec(1,4)
for i, cn in enumerate(no_comnt_school[v_features]):
    ax = plt.subplot(gs[i])
    sns.boxplot(y = cn , data = no_comnt_school)
    ax.set_title(str(cn)[0:])
    ax.set_ylabel(' ')
plt.figure(figsize=(15,8))
gs = gridspec.GridSpec(1,4)
for i, cn in enumerate(comnt_school[v_features]):
    ax = plt.subplot(gs[i])
    sns.boxplot(y = cn , data = comnt_school)
    ax.set_title(str(cn)[0:])
    ax.set_ylabel(' ')
comnt_school = School_df[School_df['Community School?'] == 'Yes']
nyccomnt_school=comnt_school[comnt_school['City'] == 'NEW YORK']
#&School_df[School_df['City'] == 'NEW YORK']
v_features = ['Percent Asian']
plt.figure(figsize=(15,8))
gs = gridspec.GridSpec(1,4)
for i, cn in enumerate(nyccomnt_school[v_features]):
    ax = plt.subplot(gs[i])
    sns.boxplot(y = cn , data =nyccomnt_school)
    ax.set_title(str(cn)[0:])
    ax.set_ylabel(' ')
v_features=['Rigorous Instruction Rating','Collaborative Teachers Rating','Supportive Environment Rating','Effective School Leadership Rating','Strong Family-Community Ties Rating','Trust Rating']
plt.figure(figsize=(20,55))
gs = gridspec.GridSpec(15, 2)
for i, cn in enumerate(School_df[v_features]):
    ax = plt.subplot(gs[i])
    sns.countplot(y=str(cn), data=School_df,order=School_df[str(cn)].value_counts().index, palette="Set2")
    ax.set_title(str(cn))
    ax.set_xlabel(' ')
    ax.set_ylabel(' ')
f,ax=plt.subplots(1,2,figsize=(15,7))
sns.barplot( y = School_df['Grade High'].dropna().value_counts().index,
            x = School_df['Grade High'].dropna().value_counts().values,
                palette="winter",ax=ax[0])
ax[0].set_title('Grade High')
ax[0].set_yticklabels(School_df['Grade High'].dropna().value_counts().index, 
                      rotation='horizontal', fontsize='large')
ax[0].set_ylabel('')
sns.barplot( y = School_df['Grade Low'].dropna().value_counts().index,
            x = School_df['Grade Low'].dropna().value_counts().values,
                palette="summer",ax=ax[1])
ax[1].set_title('Grade Low')
ax[1].set_yticklabels(School_df['Grade Low'].dropna().value_counts().index, 
                      rotation='horizontal', fontsize='large')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.8)
plt.show()