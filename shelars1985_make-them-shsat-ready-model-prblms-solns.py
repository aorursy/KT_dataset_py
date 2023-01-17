#![](https://images.fastcompany.net/image/upload/w_1280,f_auto,q_auto,fl_lossy/wp-cms/uploads/2017/05/p-1-5-credibility-busting-responses-you-need-to-stop-using.jpg)
import os
print(os.listdir("../input"))
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

School_df = pd.read_csv("../input/data-science-for-good/2016 School Explorer.csv")
School_df.tail()
layout = go.Layout(title='Top cities with no of schools', width=1000, height=500, margin=dict(l=100), xaxis=dict(tickangle=-65))
trace1 = go.Bar(x=School_df['City'].value_counts().index, y=School_df['City'].value_counts().values, marker=dict(color="#FF7441"))

data = [trace1]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
df = pd.DataFrame(School_df['Community School?'].value_counts().values,
                  index=School_df['Community School?'].value_counts().index, 
                  columns=[' '])

df.plot(kind='pie', subplots=True, autopct='%1.0f%%', figsize=(8, 8))
#plt.subplots_adjust(wspace=0.5)
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
trace0 = go.Scatter3d(
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
trace1 = go.Scatter3d(
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
      xaxis=dict(title='School Income Estimate'),
      yaxis=dict(title='Economic Need Index'),
      title=('Economic Need Assessment'))
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
#col = School_Reg_merged.columns
#y = 1
#for x in col : 
#  print (y)  
#  print(x) 
#  y = y + 1  
#v_features = School_df.iloc[:,27:38].columns
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

School_df['Grade 3 ELA 4s - Black or African American'] = School_df['Grade 3 ELA 4s - Black or African American'] / (School_df['Percent Black'] * School_df['Grade 3 ELA - All Students Tested'])
School_df['Grade 3 ELA 4s - Hispanic or Latino']  = School_df['Grade 3 ELA 4s - Hispanic or Latino'] /(School_df['Percent Hispanic']*School_df['Grade 3 ELA - All Students Tested'])
School_df['Grade 3 ELA 4s - Asian or Pacific Islander'] = School_df['Grade 3 ELA 4s - Asian or Pacific Islander'] / (School_df['Percent Asian'] * School_df['Grade 3 ELA - All Students Tested'])
School_df['Grade 3 ELA 4s - White'] = School_df['Grade 3 ELA 4s - White'] / (School_df['Percent White'] * School_df['Grade 3 ELA - All Students Tested'])
School_df['Grade 3 ELA 4s - Limited English Proficient'] = School_df['Grade 3 ELA 4s - Limited English Proficient'] / School_df['Grade 3 ELA - All Students Tested']
School_df['Grade 3 ELA 4s - Economically Disadvantaged'] = School_df['Grade 3 ELA 4s - Economically Disadvantaged'] / School_df['Grade 3 ELA - All Students Tested']
School_df['Grade 3 ELA 4s - All Students'] =School_df['Grade 3 ELA 4s - All Students'] / School_df['Grade 3 ELA - All Students Tested']

School_df['Grade 3 Math 4s - Black or African American'] = School_df['Grade 3 Math 4s - Black or African American'] / (School_df['Percent Black'] * School_df['Grade 3 Math - All Students tested'])
School_df['Grade 3 Math 4s - Hispanic or Latino']  = School_df['Grade 3 Math 4s - Hispanic or Latino'] /(School_df['Percent Hispanic']*School_df['Grade 3 Math - All Students tested'])
School_df['Grade 3 Math 4s - Asian or Pacific Islander'] = School_df['Grade 3 Math 4s - Asian or Pacific Islander'] / (School_df['Percent Asian'] * School_df['Grade 3 Math - All Students tested'])
School_df['Grade 3 Math 4s - White'] = School_df['Grade 3 Math 4s - White'] / (School_df['Percent White'] * School_df['Grade 3 Math - All Students tested'])
School_df['Grade 3 Math 4s - Limited English Proficient'] = School_df['Grade 3 Math 4s - Limited English Proficient'] / School_df['Grade 3 Math - All Students tested']
School_df['Grade 3 Math 4s - Economically Disadvantaged'] = School_df['Grade 3 Math 4s - Economically Disadvantaged'] / School_df['Grade 3 Math - All Students tested']
School_df['Grade 3 Math 4s - All Students'] =School_df['Grade 3 Math 4s - All Students'] / School_df['Grade 3 Math - All Students tested']

School_df['Grade 4 ELA 4s - Black or African American'] = School_df['Grade 4 ELA 4s - Black or African American'] / (School_df['Percent Black'] * School_df['Grade 4 ELA - All Students Tested'])
School_df['Grade 4 ELA 4s - Hispanic or Latino']  = School_df['Grade 4 ELA 4s - Hispanic or Latino'] /(School_df['Percent Hispanic']*School_df['Grade 4 ELA - All Students Tested'])
School_df['Grade 4 ELA 4s - Asian or Pacific Islander'] = School_df['Grade 4 ELA 4s - Asian or Pacific Islander'] / (School_df['Percent Asian'] * School_df['Grade 4 ELA - All Students Tested'])
School_df['Grade 4 ELA 4s - White'] = School_df['Grade 4 ELA 4s - White'] / (School_df['Percent White'] * School_df['Grade 4 ELA - All Students Tested'])
School_df['Grade 4 ELA 4s - Limited English Proficient'] = School_df['Grade 4 ELA 4s - Limited English Proficient'] / School_df['Grade 4 ELA - All Students Tested']
School_df['Grade 4 ELA 4s - Economically Disadvantaged'] = School_df['Grade 4 ELA 4s - Economically Disadvantaged'] / School_df['Grade 4 ELA - All Students Tested']
School_df['Grade 4 ELA 4s - All Students'] =School_df['Grade 4 ELA 4s - All Students'] / School_df['Grade 4 ELA - All Students Tested']


School_df['Grade 4 Math 4s - Black or African American'] = School_df['Grade 4 Math 4s - Black or African American'] / (School_df['Percent Black'] * School_df['Grade 4 Math - All Students Tested'])
School_df['Grade 4 Math 4s - Hispanic or Latino']  = School_df['Grade 4 Math 4s - Hispanic or Latino'] /(School_df['Percent Hispanic']*School_df['Grade 4 Math - All Students Tested'])
School_df['Grade 4 Math 4s - Asian or Pacific Islander'] = School_df['Grade 4 Math 4s - Asian or Pacific Islander'] / (School_df['Percent Asian'] * School_df['Grade 4 Math - All Students Tested'])
School_df['Grade 4 Math 4s - White'] = School_df['Grade 4 Math 4s - White'] / (School_df['Percent White'] * School_df['Grade 4 Math - All Students Tested'])
School_df['Grade 4 Math 4s - Limited English Proficient'] = School_df['Grade 4 Math 4s - Limited English Proficient'] / School_df['Grade 4 Math - All Students Tested']
School_df['Grade 4 Math 4s - Economically Disadvantaged'] = School_df['Grade 4 Math 4s - Economically Disadvantaged'] / School_df['Grade 4 Math - All Students Tested']
School_df['Grade 4 Math 4s - All Students'] =School_df['Grade 4 Math 4s - All Students'] / School_df['Grade 4 Math - All Students Tested']

School_df['Grade 5 ELA 4s - Black or African American'] = School_df['Grade 5 ELA 4s - Black or African American'] / (School_df['Percent Black'] * School_df['Grade 5 ELA - All Students Tested'])
School_df['Grade 5 ELA 4s - Hispanic or Latino']  = School_df['Grade 5 ELA 4s - Hispanic or Latino'] /(School_df['Percent Hispanic']*School_df['Grade 5 ELA - All Students Tested'])
School_df['Grade 5 ELA 4s - Asian or Pacific Islander'] = School_df['Grade 5 ELA 4s - Asian or Pacific Islander'] / (School_df['Percent Asian'] * School_df['Grade 5 ELA - All Students Tested'])
School_df['Grade 5 ELA 4s - White'] = School_df['Grade 5 ELA 4s - White'] / (School_df['Percent White'] * School_df['Grade 5 ELA - All Students Tested'])
School_df['Grade 5 ELA 4s - Limited English Proficient'] = School_df['Grade 5 ELA 4s - Limited English Proficient'] / School_df['Grade 5 ELA - All Students Tested']
School_df['Grade 5 ELA 4s - Economically Disadvantaged'] = School_df['Grade 5 ELA 4s - Economically Disadvantaged'] / School_df['Grade 5 ELA - All Students Tested']
School_df['Grade 5 ELA 4s - All Students'] =School_df['Grade 5 ELA 4s - All Students'] / School_df['Grade 5 ELA - All Students Tested']


School_df['Grade 5 Math 4s - Black or African American'] = School_df['Grade 5 Math 4s - Black or African American'] / (School_df['Percent Black'] * School_df['Grade 5 Math - All Students Tested'])
School_df['Grade 5 Math 4s - Hispanic or Latino']  = School_df['Grade 5 Math 4s - Hispanic or Latino'] /(School_df['Percent Hispanic']*School_df['Grade 5 Math - All Students Tested'])
School_df['Grade 5 Math 4s - Asian or Pacific Islander'] = School_df['Grade 5 Math 4s - Asian or Pacific Islander'] / (School_df['Percent Asian'] * School_df['Grade 5 Math - All Students Tested'])
School_df['Grade 5 Math 4s - White'] = School_df['Grade 5 Math 4s - White'] / (School_df['Percent White'] * School_df['Grade 5 Math - All Students Tested'])
School_df['Grade 5 Math 4s - Limited English Proficient'] = School_df['Grade 5 Math 4s - Limited English Proficient'] / School_df['Grade 5 Math - All Students Tested']
School_df['Grade 5 Math 4s - Economically Disadvantaged'] = School_df['Grade 5 Math 4s - Economically Disadvantaged'] / School_df['Grade 5 Math - All Students Tested']
School_df['Grade 5 Math 4s - All Students'] =School_df['Grade 5 Math 4s - All Students'] / School_df['Grade 5 Math - All Students Tested']


School_df['Grade 6 ELA 4s - Black or African American'] = School_df['Grade 6 ELA 4s - Black or African American'] / (School_df['Percent Black'] * School_df['Grade 6 ELA - All Students Tested'])
School_df['Grade 6 ELA 4s - Hispanic or Latino']  = School_df['Grade 6 ELA 4s - Hispanic or Latino'] /(School_df['Percent Hispanic']*School_df['Grade 6 ELA - All Students Tested'])
School_df['Grade 6 ELA 4s - Asian or Pacific Islander'] = School_df['Grade 6 ELA 4s - Asian or Pacific Islander'] / (School_df['Percent Asian'] * School_df['Grade 6 ELA - All Students Tested'])
School_df['Grade 6 ELA 4s - White'] = School_df['Grade 6 ELA 4s - White'] / (School_df['Percent White'] * School_df['Grade 6 ELA - All Students Tested'])
School_df['Grade 6 ELA 4s - Limited English Proficient'] = School_df['Grade 6 ELA 4s - Limited English Proficient'] / School_df['Grade 6 ELA - All Students Tested']
School_df['Grade 6 ELA 4s - Economically Disadvantaged'] = School_df['Grade 6 ELA 4s - Economically Disadvantaged'] / School_df['Grade 6 ELA - All Students Tested']
School_df['Grade 6 ELA 4s - All Students'] =School_df['Grade 6 ELA 4s - All Students'] / School_df['Grade 6 ELA - All Students Tested']

School_df['Grade 6 Math 4s - Black or African American'] = School_df['Grade 6 Math 4s - Black or African American'] / (School_df['Percent Black'] * School_df['Grade 6 Math - All Students Tested'])
School_df['Grade 6 Math 4s - Hispanic or Latino']  = School_df['Grade 6 Math 4s - Hispanic or Latino'] /(School_df['Percent Hispanic']*School_df['Grade 6 Math - All Students Tested'])
School_df['Grade 6 Math 4s - Asian or Pacific Islander'] = School_df['Grade 6 Math 4s - Asian or Pacific Islander'] / (School_df['Percent Asian'] * School_df['Grade 6 Math - All Students Tested'])
School_df['Grade 6 Math 4s - White'] = School_df['Grade 6 Math 4s - White'] / (School_df['Percent White'] * School_df['Grade 6 Math - All Students Tested'])
School_df['Grade 6 Math 4s - Limited English Proficient'] = School_df['Grade 6 Math 4s - Limited English Proficient'] / School_df['Grade 6 Math - All Students Tested']
School_df['Grade 6 Math 4s - Economically Disadvantaged'] = School_df['Grade 6 Math 4s - Economically Disadvantaged'] / School_df['Grade 6 Math - All Students Tested']
School_df['Grade 6 Math 4s - All Students'] =School_df['Grade 6 Math 4s - All Students'] / School_df['Grade 6 Math - All Students Tested']

School_df['Grade 7 ELA 4s - Black or African American'] = School_df['Grade 7 ELA 4s - Black or African American'] / (School_df['Percent Black'] * School_df['Grade 7 ELA - All Students Tested'])
School_df['Grade 7 ELA 4s - Hispanic or Latino']  = School_df['Grade 7 ELA 4s - Hispanic or Latino'] /(School_df['Percent Hispanic']*School_df['Grade 7 ELA - All Students Tested'])
School_df['Grade 7 ELA 4s - Asian or Pacific Islander'] = School_df['Grade 7 ELA 4s - Asian or Pacific Islander'] / (School_df['Percent Asian'] * School_df['Grade 7 ELA - All Students Tested'])
School_df['Grade 7 ELA 4s - White'] = School_df['Grade 7 ELA 4s - White'] / (School_df['Percent White'] * School_df['Grade 7 ELA - All Students Tested'])
School_df['Grade 7 ELA 4s - Limited English Proficient'] = School_df['Grade 7 ELA 4s - Limited English Proficient'] / School_df['Grade 7 ELA - All Students Tested']
School_df['Grade 7 ELA 4s - Economically Disadvantaged'] = School_df['Grade 7 ELA 4s - Economically Disadvantaged'] / School_df['Grade 7 ELA - All Students Tested']
School_df['Grade 7 ELA 4s - All Students'] =School_df['Grade 7 ELA 4s - All Students'] / School_df['Grade 7 ELA - All Students Tested']

School_df['Grade 7 Math 4s - Black or African American'] = School_df['Grade 7 Math 4s - Black or African American'] / (School_df['Percent Black'] * School_df['Grade 7 Math - All Students Tested'])
School_df['Grade 7 Math 4s - Hispanic or Latino']  = School_df['Grade 7 Math 4s - Hispanic or Latino'] /(School_df['Percent Hispanic']*School_df['Grade 7 Math - All Students Tested'])
School_df['Grade 7 Math 4s - Asian or Pacific Islander'] = School_df['Grade 7 Math 4s - Asian or Pacific Islander'] / (School_df['Percent Asian'] * School_df['Grade 7 Math - All Students Tested'])
School_df['Grade 7 Math 4s - White'] = School_df['Grade 7 Math 4s - White'] / (School_df['Percent White'] * School_df['Grade 7 Math - All Students Tested'])
School_df['Grade 7 Math 4s - Limited English Proficient'] = School_df['Grade 7 Math 4s - Limited English Proficient'] / School_df['Grade 7 Math - All Students Tested']
School_df['Grade 7 Math 4s - Economically Disadvantaged'] = School_df['Grade 7 Math 4s - Economically Disadvantaged'] / School_df['Grade 7 Math - All Students Tested']
School_df['Grade 7 Math 4s - All Students'] =School_df['Grade 7 Math 4s - All Students'] / School_df['Grade 7 Math - All Students Tested']

School_df['Grade 8 ELA 4s - Black or African American'] = School_df['Grade 8 ELA 4s - Black or African American'] / (School_df['Percent Black'] * School_df['Grade 8 ELA - All Students Tested'])
School_df['Grade 8 ELA 4s - Hispanic or Latino']  = School_df['Grade 8 ELA 4s - Hispanic or Latino'] /(School_df['Percent Hispanic']*School_df['Grade 8 ELA - All Students Tested'])
School_df['Grade 8 ELA 4s - Asian or Pacific Islander'] = School_df['Grade 8 ELA 4s - Asian or Pacific Islander'] / (School_df['Percent Asian'] * School_df['Grade 8 ELA - All Students Tested'])
School_df['Grade 8 ELA 4s - White'] = School_df['Grade 8 ELA 4s - White'] / (School_df['Percent White'] * School_df['Grade 8 ELA - All Students Tested'])
School_df['Grade 8 ELA 4s - Limited English Proficient'] = School_df['Grade 8 ELA 4s - Limited English Proficient'] / School_df['Grade 8 ELA - All Students Tested']
School_df['Grade 8 ELA 4s - Economically Disadvantaged'] = School_df['Grade 8 ELA 4s - Economically Disadvantaged'] / School_df['Grade 8 ELA - All Students Tested']
School_df['Grade 8 ELA 4s - All Students'] =School_df['Grade 8 ELA 4s - All Students'] / School_df['Grade 8 ELA - All Students Tested']

School_df['Grade 8 Math 4s - Black or African American'] = School_df['Grade 8 Math 4s - Black or African American'] / (School_df['Percent Black'] * School_df['Grade 8 Math - All Students Tested'])
School_df['Grade 8 Math 4s - Hispanic or Latino']  = School_df['Grade 8 Math 4s - Hispanic or Latino'] /(School_df['Percent Hispanic']*School_df['Grade 8 Math - All Students Tested'])
School_df['Grade 8 Math 4s - Asian or Pacific Islander'] = School_df['Grade 8 Math 4s - Asian or Pacific Islander'] / (School_df['Percent Asian'] * School_df['Grade 8 Math - All Students Tested'])
School_df['Grade 8 Math 4s - White'] = School_df['Grade 8 Math 4s - White'] / (School_df['Percent White'] * School_df['Grade 8 Math - All Students Tested'])
School_df['Grade 8 Math 4s - Limited English Proficient'] = School_df['Grade 8 Math 4s - Limited English Proficient'] / School_df['Grade 8 Math - All Students Tested']
School_df['Grade 8 Math 4s - Economically Disadvantaged'] = School_df['Grade 8 Math 4s - Economically Disadvantaged'] / School_df['Grade 8 Math - All Students Tested']
School_df['Grade 8 Math 4s - All Students'] =School_df['Grade 8 Math 4s - All Students'] / School_df['Grade 8 Math - All Students Tested']

School_df['Grade 3 ELA 4s - Black or African American'].fillna(0, inplace=True) 
School_df['Grade 3 ELA 4s - Hispanic or Latino'].fillna(0, inplace=True)  
School_df['Grade 3 ELA 4s - Asian or Pacific Islander'].fillna(0, inplace=True) 
School_df['Grade 3 ELA 4s - White'].fillna(0, inplace=True) 
School_df['Grade 3 ELA 4s - Limited English Proficient'].fillna(0, inplace=True) 
School_df['Grade 3 ELA 4s - Economically Disadvantaged'].fillna(0, inplace=True) 
School_df['Grade 3 ELA 4s - All Students'].fillna(0, inplace=True) 

School_df['Grade 3 Math 4s - Black or African American'].fillna(0, inplace=True) 
School_df['Grade 3 Math 4s - Hispanic or Latino'].fillna(0, inplace=True)  
School_df['Grade 3 Math 4s - Asian or Pacific Islander'].fillna(0, inplace=True) 
School_df['Grade 3 Math 4s - White'].fillna(0, inplace=True) 
School_df['Grade 3 Math 4s - Limited English Proficient'].fillna(0, inplace=True) 
School_df['Grade 3 Math 4s - Economically Disadvantaged'].fillna(0, inplace=True) 
School_df['Grade 3 Math 4s - All Students'].fillna(0, inplace=True) 

School_df['Grade 4 ELA 4s - Black or African American'].fillna(0, inplace=True) 
School_df['Grade 4 ELA 4s - Hispanic or Latino'].fillna(0, inplace=True)  
School_df['Grade 4 ELA 4s - Asian or Pacific Islander'].fillna(0, inplace=True) 
School_df['Grade 4 ELA 4s - White'].fillna(0, inplace=True) 
School_df['Grade 4 ELA 4s - Limited English Proficient'].fillna(0, inplace=True) 
School_df['Grade 4 ELA 4s - Economically Disadvantaged'].fillna(0, inplace=True) 
School_df['Grade 4 ELA 4s - All Students'].fillna(0, inplace=True) 

School_df['Grade 4 Math 4s - Black or African American'].fillna(0, inplace=True) 
School_df['Grade 4 Math 4s - Hispanic or Latino'].fillna(0, inplace=True)  
School_df['Grade 4 Math 4s - Asian or Pacific Islander'].fillna(0, inplace=True) 
School_df['Grade 4 Math 4s - White'].fillna(0, inplace=True) 
School_df['Grade 4 Math 4s - Limited English Proficient'].fillna(0, inplace=True) 
School_df['Grade 4 Math 4s - Economically Disadvantaged'].fillna(0, inplace=True) 
School_df['Grade 4 Math 4s - All Students'].fillna(0, inplace=True) 

School_df['Grade 5 ELA 4s - Black or African American'].fillna(0, inplace=True) 
School_df['Grade 5 ELA 4s - Hispanic or Latino'].fillna(0, inplace=True)  
School_df['Grade 5 ELA 4s - Asian or Pacific Islander'].fillna(0, inplace=True) 
School_df['Grade 5 ELA 4s - White'].fillna(0, inplace=True) 
School_df['Grade 5 ELA 4s - Limited English Proficient'].fillna(0, inplace=True) 
School_df['Grade 5 ELA 4s - Economically Disadvantaged'].fillna(0, inplace=True) 
School_df['Grade 5 ELA 4s - All Students'].fillna(0, inplace=True) 

School_df['Grade 5 Math 4s - Black or African American'].fillna(0, inplace=True) 
School_df['Grade 5 Math 4s - Hispanic or Latino'].fillna(0, inplace=True)  
School_df['Grade 5 Math 4s - Asian or Pacific Islander'].fillna(0, inplace=True) 
School_df['Grade 5 Math 4s - White'].fillna(0, inplace=True) 
School_df['Grade 5 Math 4s - Limited English Proficient'].fillna(0, inplace=True) 
School_df['Grade 5 Math 4s - Economically Disadvantaged'].fillna(0, inplace=True) 
School_df['Grade 5 Math 4s - All Students'].fillna(0, inplace=True) 

School_df['Grade 6 ELA 4s - Black or African American'].fillna(0, inplace=True) 
School_df['Grade 6 ELA 4s - Hispanic or Latino'].fillna(0, inplace=True)  
School_df['Grade 6 ELA 4s - Asian or Pacific Islander'].fillna(0, inplace=True) 
School_df['Grade 6 ELA 4s - White'].fillna(0, inplace=True) 
School_df['Grade 6 ELA 4s - Limited English Proficient'].fillna(0, inplace=True) 
School_df['Grade 6 ELA 4s - Economically Disadvantaged'].fillna(0, inplace=True) 
School_df['Grade 6 ELA 4s - All Students'].fillna(0, inplace=True) 

School_df['Grade 6 Math 4s - Black or African American'].fillna(0, inplace=True) 
School_df['Grade 6 Math 4s - Hispanic or Latino'].fillna(0, inplace=True)  
School_df['Grade 6 Math 4s - Asian or Pacific Islander'].fillna(0, inplace=True) 
School_df['Grade 6 Math 4s - White'].fillna(0, inplace=True) 
School_df['Grade 6 Math 4s - Limited English Proficient'].fillna(0, inplace=True) 
School_df['Grade 6 Math 4s - Economically Disadvantaged'].fillna(0, inplace=True) 
School_df['Grade 6 Math 4s - All Students'].fillna(0, inplace=True) 

School_df['Grade 7 ELA 4s - Black or African American'].fillna(0, inplace=True) 
School_df['Grade 7 ELA 4s - Hispanic or Latino'].fillna(0, inplace=True)  
School_df['Grade 7 ELA 4s - Asian or Pacific Islander'].fillna(0, inplace=True) 
School_df['Grade 7 ELA 4s - White'].fillna(0, inplace=True) 
School_df['Grade 7 ELA 4s - Limited English Proficient'].fillna(0, inplace=True) 
School_df['Grade 7 ELA 4s - Economically Disadvantaged'].fillna(0, inplace=True) 
School_df['Grade 7 ELA 4s - All Students'].fillna(0, inplace=True) 

School_df['Grade 7 Math 4s - Black or African American'].fillna(0, inplace=True) 
School_df['Grade 7 Math 4s - Hispanic or Latino'].fillna(0, inplace=True)  
School_df['Grade 7 Math 4s - Asian or Pacific Islander'].fillna(0, inplace=True) 
School_df['Grade 7 Math 4s - White'].fillna(0, inplace=True) 
School_df['Grade 7 Math 4s - Limited English Proficient'].fillna(0, inplace=True) 
School_df['Grade 7 Math 4s - Economically Disadvantaged'].fillna(0, inplace=True) 
School_df['Grade 7 Math 4s - All Students'].fillna(0, inplace=True) 

School_df['Grade 8 ELA 4s - Black or African American'].fillna(0, inplace=True) 
School_df['Grade 8 ELA 4s - Hispanic or Latino'].fillna(0, inplace=True)  
School_df['Grade 8 ELA 4s - Asian or Pacific Islander'].fillna(0, inplace=True) 
School_df['Grade 8 ELA 4s - White'].fillna(0, inplace=True) 
School_df['Grade 8 ELA 4s - Limited English Proficient'].fillna(0, inplace=True) 
School_df['Grade 8 ELA 4s - Economically Disadvantaged'].fillna(0, inplace=True) 
School_df['Grade 8 ELA 4s - All Students'].fillna(0, inplace=True) 

School_df['Grade 8 Math 4s - Black or African American'].fillna(0, inplace=True) 
School_df['Grade 8 Math 4s - Hispanic or Latino'].fillna(0, inplace=True)  
School_df['Grade 8 Math 4s - Asian or Pacific Islander'].fillna(0, inplace=True) 
School_df['Grade 8 Math 4s - White'].fillna(0, inplace=True) 
School_df['Grade 8 Math 4s - Limited English Proficient'].fillna(0, inplace=True) 
School_df['Grade 8 Math 4s - Economically Disadvantaged'].fillna(0, inplace=True) 
School_df['Grade 8 Math 4s - All Students'].fillna(0, inplace=True) 

ELA = {'Race': ['Black', 'Hispanic', 'Asian','White'], 
           'G3': [School_df['Grade 3 ELA 4s - Black or African American'].mean(), School_df['Grade 3 ELA 4s - Hispanic or Latino'].mean(), School_df['Grade 3 ELA 4s - Asian or Pacific Islander'].mean(),School_df['Grade 3 ELA 4s - White'].mean()],
           'G4': [School_df['Grade 4 ELA 4s - Black or African American'].mean(), School_df['Grade 4 ELA 4s - Hispanic or Latino'].mean(), School_df['Grade 4 ELA 4s - Asian or Pacific Islander'].mean(),School_df['Grade 4 ELA 4s - White'].mean()],
           'G5': [School_df['Grade 5 ELA 4s - Black or African American'].mean(), School_df['Grade 5 ELA 4s - Hispanic or Latino'].mean(), School_df['Grade 5 ELA 4s - Asian or Pacific Islander'].mean(),School_df['Grade 5 ELA 4s - White'].mean()],
           'G6': [School_df['Grade 6 ELA 4s - Black or African American'].mean(), School_df['Grade 6 ELA 4s - Hispanic or Latino'].mean(), School_df['Grade 6 ELA 4s - Asian or Pacific Islander'].mean(),School_df['Grade 6 ELA 4s - White'].mean()],
           'G7': [School_df['Grade 7 ELA 4s - Black or African American'].mean(), School_df['Grade 7 ELA 4s - Hispanic or Latino'].mean(), School_df['Grade 7 ELA 4s - Asian or Pacific Islander'].mean(),School_df['Grade 7 ELA 4s - White'].mean()],
           'G8': [School_df['Grade 8 ELA 4s - Black or African American'].mean(), School_df['Grade 8 ELA 4s - Hispanic or Latino'].mean(), School_df['Grade 8 ELA 4s - Asian or Pacific Islander'].mean(),School_df['Grade 8 ELA 4s - White'].mean()]}

index = [0,1,2,3]
ELA_df = pd.DataFrame(ELA, index=index)

Math = {'Race': ['Black', 'Hispanic', 'Asian','White'], 
           'G3': [School_df['Grade 3 Math 4s - Black or African American'].mean(), School_df['Grade 3 Math 4s - Hispanic or Latino'].mean(), School_df['Grade 3 Math 4s - Asian or Pacific Islander'].mean(),School_df['Grade 3 Math 4s - White'].mean()],
           'G4': [School_df['Grade 4 Math 4s - Black or African American'].mean(), School_df['Grade 4 Math 4s - Hispanic or Latino'].mean(), School_df['Grade 4 Math 4s - Asian or Pacific Islander'].mean(),School_df['Grade 4 Math 4s - White'].mean()],
           'G5': [School_df['Grade 5 Math 4s - Black or African American'].mean(), School_df['Grade 5 Math 4s - Hispanic or Latino'].mean(), School_df['Grade 5 Math 4s - Asian or Pacific Islander'].mean(),School_df['Grade 5 Math 4s - White'].mean()],
           'G6': [School_df['Grade 6 Math 4s - Black or African American'].mean(), School_df['Grade 6 Math 4s - Hispanic or Latino'].mean(), School_df['Grade 6 Math 4s - Asian or Pacific Islander'].mean(),School_df['Grade 6 Math 4s - White'].mean()],
           'G7': [School_df['Grade 7 Math 4s - Black or African American'].mean(), School_df['Grade 7 Math 4s - Hispanic or Latino'].mean(), School_df['Grade 7 Math 4s - Asian or Pacific Islander'].mean(),School_df['Grade 7 Math 4s - White'].mean()],
           'G8': [School_df['Grade 8 Math 4s - Black or African American'].mean(), School_df['Grade 8 Math 4s - Hispanic or Latino'].mean(), School_df['Grade 8 Math 4s - Asian or Pacific Islander'].mean(),School_df['Grade 8 Math 4s - White'].mean()]}

index = [0,1,2,3]
Math_df1 = pd.DataFrame(Math, index=index)
trace0 = go.Scatter(
    x = Math_df1['Race'],y = Math_df1['G3'],mode='lines',name='Grade 3 students scoring in level 4 - Math ')
trace1 = go.Scatter(
    x = Math_df1['Race'],y = Math_df1['G4'],mode='lines',name='Grade 4 students scoring in level 4 - Math ')
trace2 = go.Scatter(
    x = Math_df1['Race'],y = Math_df1['G5'],mode='lines',name='Grade 5 students scoring in level 4 - Math ')
trace3 = go.Scatter(
    x = Math_df1['Race'],y = Math_df1['G6'],mode='lines',name='Grade 6 students scoring in level 4 - Math ')
trace4 = go.Scatter(
    x = Math_df1['Race'],y = Math_df1['G7'],mode='lines',name='Grade 7 students scoring in level 4 - Math ')
trace5 = go.Scatter(
    x = Math_df1['Race'],y = Math_df1['G8'],mode='lines',name='Grade 8 students scoring in level 4 - Math ')


data = [trace0,trace1,trace2,trace3,trace4,trace5]

layout = go.Layout(
    xaxis=dict(title='Race'),
    yaxis=dict(title='Percent students from each race'),
    title=' Students performance in different grade - Math ',
    showlegend = True)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

trace0 = go.Scatter(
    x = ELA_df['Race'],y = ELA_df['G3'],mode='lines',name='Grade 3 students scoring in level 4 - ELA ')
trace1 = go.Scatter(
    x = ELA_df['Race'],y = ELA_df['G4'],mode='lines',name='Grade 4 students scoring in level 4 - ELA ')
trace2 = go.Scatter(
    x = ELA_df['Race'],y = ELA_df['G5'],mode='lines',name='Grade 5 students scoring in level 4 - ELA ')
trace3 = go.Scatter(
    x = ELA_df['Race'],y = ELA_df['G6'],mode='lines',name='Grade 6 students scoring in level 4 - ELA ')
trace4 = go.Scatter(
    x = ELA_df['Race'],y = ELA_df['G7'],mode='lines',name='Grade 7 students scoring in level 4 - ELA ')
trace5 = go.Scatter(
    x = ELA_df['Race'],y = ELA_df['G8'],mode='lines',name='Grade 8 students scoring in level 4 - ELA ')


data = [trace0,trace1,trace2,trace3,trace4,trace5]

layout = go.Layout(
    xaxis=dict(title='Race'),
    yaxis=dict(title='Percent students from each race'),
    title=' Students performance in different grade - ELA ',
    showlegend = True)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

v_features = ['Grade 3 ELA - All Students Tested','Grade 3 Math - All Students tested',
              'Grade 4 ELA - All Students Tested','Grade 4 Math - All Students Tested',
              'Grade 5 ELA - All Students Tested','Grade 5 Math - All Students Tested',
              'Grade 6 ELA - All Students Tested','Grade 6 Math - All Students Tested',
              'Grade 7 ELA - All Students Tested','Grade 7 Math - All Students Tested',
              'Grade 8 ELA - All Students Tested','Grade 8 Math - All Students Tested',]
plt.figure(figsize=(12,8*4))
gs = gridspec.GridSpec(7, 2)
for i, cn in enumerate(School_df[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(School_df[cn], bins=5)
    ax.set_xlabel('')
    ax.set_title('feature: ' + str(cn))
plt.show()
def get_bins(no):
    if no == 0 :
        return 0
    elif no > 0 and no <= 50 :
        return 1
    elif no > 50 and no <= 100 :
        return 2
    elif no > 100 and no <= 150 :
        return 3
    elif no > 150 and no <= 200 :
        return 4
    else: 
        return 5

v_features = ['Grade 3 ELA - All Students Tested','Grade 3 Math - All Students tested',
              'Grade 4 ELA - All Students Tested','Grade 4 Math - All Students Tested',
              'Grade 5 ELA - All Students Tested','Grade 5 Math - All Students Tested',
              'Grade 6 ELA - All Students Tested','Grade 6 Math - All Students Tested',
              'Grade 7 ELA - All Students Tested','Grade 7 Math - All Students Tested',
              'Grade 8 ELA - All Students Tested','Grade 8 Math - All Students Tested',]
for i, cn in enumerate(School_df[v_features]):
    School_df[cn] = School_df[cn].apply(lambda x: get_bins(x))
plt.figure(figsize=(12,8*4))
gs = gridspec.GridSpec(7, 2)
for i, cn in enumerate(School_df[v_features]):
    ax = plt.subplot(gs[i])
    sns.countplot(y=str(cn), data=School_df,order=School_df[str(cn)].value_counts().index, palette="Set2")
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('feature: ' + str(cn))
plt.show()
School_df['Community School?'] = School_df['Community School?'].map(lambda s: 1 if s == 'Yes' else 0)

School_df['Percent ELL']=School_df['Percent ELL'].replace({'%':'', ',':''},regex=True).astype(float)
features = ["Student Attendance Rate","Percent of Students Chronically Absent","Rigorous Instruction %"
,"Collaborative Teachers %","Supportive Environment %","Effective School Leadership %","Strong Family-Community Ties %","Trust %",]

for i, cn in enumerate(School_df[features]):
     School_df[str(cn)]= (School_df[str(cn)].replace({'%':'', ',':''},regex=True).astype(float)/100)
School_df['Rigorous Instruction Rating'].fillna(0, inplace=True)
School_df['Collaborative Teachers Rating'].fillna(0, inplace=True)
School_df['Supportive Environment Rating'].fillna(0, inplace=True)
School_df['Effective School Leadership Rating'].fillna(0, inplace=True)
School_df['Strong Family-Community Ties Rating'].fillna(0, inplace=True)
School_df['Trust Rating'].fillna(0, inplace=True)
School_df['Student Achievement Rating'].fillna(0, inplace=True)

School_df['Rigorous Instruction Rating'] = School_df['Rigorous Instruction Rating'].map({"Not Meeting Target":0,"Approaching Target":1, "Meeting Target":2, "Exceeding Target" : 3, 0 : 0})
School_df['Rigorous Instruction Rating'] = School_df['Rigorous Instruction Rating'].astype(int)
School_df['Collaborative Teachers Rating'] = School_df['Collaborative Teachers Rating'].map({"Not Meeting Target":0,"Approaching Target":1, "Meeting Target":2, "Exceeding Target" : 3, 0 : 0})
School_df['Collaborative Teachers Rating'] = School_df['Collaborative Teachers Rating'].astype(int)
School_df['Supportive Environment Rating'] = School_df['Supportive Environment Rating'].map({"Not Meeting Target":0,"Approaching Target":1, "Meeting Target":2, "Exceeding Target" : 3, 0 : 0})
School_df['Supportive Environment Rating'] = School_df['Supportive Environment Rating'].astype(int)
School_df['Effective School Leadership Rating'] = School_df['Effective School Leadership Rating'].map({"Not Meeting Target":0,"Approaching Target":1, "Meeting Target":2, "Exceeding Target" : 3, 0 : 0})
School_df['Effective School Leadership Rating'] = School_df['Effective School Leadership Rating'].astype(int)
School_df['Strong Family-Community Ties Rating'] = School_df['Strong Family-Community Ties Rating'].map({"Not Meeting Target":0,"Approaching Target":1, "Meeting Target":2, "Exceeding Target" : 3, 0 : 0})
School_df['Strong Family-Community Ties Rating'] = School_df['Strong Family-Community Ties Rating'].astype(int)
School_df['Trust Rating'] = School_df['Trust Rating'].map({"Not Meeting Target":0,"Approaching Target":1, "Meeting Target":2, "Exceeding Target" : 3, 0 : 0})
School_df['Trust Rating'] = School_df['Trust Rating'].astype(int)
School_df['Student Achievement Rating'] = School_df['Student Achievement Rating'].map({"Not Meeting Target":0,"Approaching Target":1, "Meeting Target":2, "Exceeding Target" : 3, 0 : 0})
School_df['Student Achievement Rating'] = School_df['Student Achievement Rating'].astype(int)
Registration_df = pd.read_csv("../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv")
Registration_df["percent1"] = Registration_df["Number of students who took the SHSAT"]/Registration_df["Number of students who registered for the SHSAT"]
Registration_df["percent2"] = Registration_df["Number of students who registered for the SHSAT"]/Registration_df["Enrollment on 10/31"]
Registration_df["Reg_idx"] = Registration_df["percent1"]*Registration_df["percent2"]
Registration_df = Registration_df.drop_duplicates(subset=['School name','Year of SHST'])
Registration_df.head()
Registration_df1 = Registration_df.groupby(['School name','Year of SHST'])['percent2'].sum().unstack()
Registration_df1 = Registration_df1.sort_values([2016], ascending=False)
Registration_df1 = Registration_df1.fillna(0)
f, ax = plt.subplots(figsize=(15, 10)) 
g = sns.heatmap(Registration_df1,annot=True,cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
Registration_df2 = Registration_df.groupby(['School name','Year of SHST'])['percent1'].sum().unstack()
Registration_df2 = Registration_df2.sort_values([2016], ascending=False)
Registration_df2 = Registration_df2.fillna(0)
f, ax = plt.subplots(figsize=(15, 10)) 
g = sns.heatmap(Registration_df2,annot=True,cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
Registration_df3 = Registration_df.drop_duplicates(subset=['School name','Year of SHST'])
Registration_df4 = Registration_df3.groupby(['School name','Year of SHST'])['Reg_idx'].sum().unstack()
Registration_df4 = Registration_df4.sort_values([2016], ascending=False)
Registration_df4 = Registration_df4.fillna(0)
f, ax = plt.subplots(figsize=(15, 10)) 
g = sns.heatmap(Registration_df4,annot=True,cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show();
filtered_reg_df = Registration_df.drop_duplicates(subset=['School name','Year of SHST'])
filtered_reg_df = filtered_reg_df[filtered_reg_df['Year of SHST'] == 2016] 
filtered_reg_df = filtered_reg_df[filtered_reg_df['Grade level'] == 8]
School_Reg_merged = pd.merge(filtered_reg_df, School_df, how='left', left_on='DBN', right_on='Location Code')
School_Reg_merged = School_Reg_merged[np.isfinite(School_Reg_merged['Economic Need Index'])]
School_Reg_merged.head()
School_Reg_merged['Grade level'].value_counts()
fig, ax = plt.subplots(figsize=(13, 13))
ax.scatter(School_Reg_merged['Economic Need Index'],School_Reg_merged['Reg_idx'],marker="o", color="lightBlue", s=10, linewidths=10)
ax.set_xlabel('Economic Need Index')
ax.set_ylabel('percent participation in 2016')
ax.spines['right'].set_visible(False)
ax.grid()
plt.grid()

ENI = School_Reg_merged['Economic Need Index']
ENI = np.array(ENI)
school = School_Reg_merged['School name']
school = np.array(school)
percent = School_Reg_merged['Reg_idx']
percent = np.array(percent)

for i, txt in enumerate(school):
      ax.annotate(txt, (ENI[i],percent[i]),fontsize=12,rotation=-15,color='Red')

ax.annotate('Threshold', (0.57,0.20),fontsize=14,rotation=0,color='Blue')
ax.annotate('.', xy=(0.59,0.21), xytext=(0.59, 0.25),
            arrowprops=dict(facecolor='Red', shrink=0.06),)
ax.annotate('.', xy=(0.59,0.19), xytext=(0.59, 0.15),
            arrowprops=dict(facecolor='Red', shrink=0.06),);
        
School_Reg_merged['Reg_idx'] = School_Reg_merged['Reg_idx'].map(lambda s: 1 if s >= 0.20 else 0)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
School_Reg_merged.drop(['School Income Estimate','Adjusted Grade','New?','Other Location Code in LCGMS'], axis =1, inplace = True)
rnd_clf = RandomForestClassifier(n_estimators = 100 , criterion = 'entropy',random_state = 0)
rnd_clf.fit(School_Reg_merged.iloc[:,22:167],School_Reg_merged.iloc[:,9]);
#for name, importance in zip(School_Reg_merged.iloc[:,22:167].columns, rnd_clf.feature_importances_):
#    print(name, "=", importance)
'''
plt.figure(figsize=(12,8*4))
g = sns.barplot(y=School_Reg_merged.iloc[:,22:167].columns,x = rnd_clf.feature_importances_, orient='h')

'''

x, y = (list(x) for x in zip(*sorted(zip(rnd_clf.feature_importances_, School_Reg_merged.iloc[:,22:167].columns), 
                                                            reverse = False)))
trace2 = go.Bar(
    x=x ,
    y=y,
    marker=dict(
        color=x,
        colorscale = 'Viridis',
        reversescale = True
    ),
    name='Random Forest Feature importance',
    orientation='h',
)

layout = dict(
    title='Barplot of Feature importances',
     width = 900, height = 3000,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
#         domain=[0, 0.85],
    ),
    margin=dict(
    l=300,
),
)

fig1 = go.Figure(data=[trace2], layout=layout)
iplot(fig1)

for name, importance in zip(School_Reg_merged.iloc[:,22:167].columns, rnd_clf.feature_importances_):
    if importance > 0.015 :
        print('"' + name + '"'+',')
School_Reg_train = School_Reg_merged[["School name",
"Percent Black / Hispanic",
"Student Attendance Rate",
"Percent of Students Chronically Absent",
"Rigorous Instruction %",
"Collaborative Teachers %",
"Supportive Environment %",
"Effective School Leadership %",
"Strong Family-Community Ties %",
"Trust %",
"Student Achievement Rating",
"Average ELA Proficiency",
"Grade 5 ELA 4s - All Students",
"Grade 5 ELA 4s - Economically Disadvantaged",
"Grade 6 ELA 4s - All Students",
"Grade 6 ELA 4s - Black or African American",
"Grade 6 ELA 4s - Hispanic or Latino",
"Grade 6 Math 4s - All Students",
"Grade 6 Math 4s - Hispanic or Latino",
"Grade 6 Math 4s - Economically Disadvantaged",
"Grade 7 ELA 4s - All Students",
"Grade 8 ELA - All Students Tested",
"Grade 8 ELA 4s - All Students",
"Grade 8 ELA 4s - Hispanic or Latino",
"Grade 8 ELA 4s - Economically Disadvantaged",
"Grade 8 Math 4s - Economically Disadvantaged",
"Reg_idx"]]
X = School_Reg_train.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]].values 
y = School_Reg_train.iloc[:, 26].values

kfold = StratifiedKFold(n_splits=9)
random_state = 2
classifiers = []
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(LogisticRegression(random_state = random_state))

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X, y = y, scoring = "accuracy", cv = kfold, n_jobs=4))
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["RandomForest","LogisticRegression"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
RFC = RandomForestClassifier()
## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[300,1000],
              "criterion": ["gini"]}

gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=9, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X,y)
RFC_best = gsRFC.best_estimator_
# Best score
gsRFC.best_score_
School_Reg = School_df[['School Name',
'Grade High','Percent White','Percent Asian','Latitude','Longitude',
"Percent Black / Hispanic",
"Student Attendance Rate",
"Percent of Students Chronically Absent",
"Rigorous Instruction %",
"Collaborative Teachers %",
"Supportive Environment %",
"Effective School Leadership %",
"Strong Family-Community Ties %",
"Trust %",
"Student Achievement Rating",
"Average ELA Proficiency",
"Grade 5 ELA 4s - All Students",
"Grade 5 ELA 4s - Economically Disadvantaged",
"Grade 6 ELA 4s - All Students",
"Grade 6 ELA 4s - Black or African American",
"Grade 6 ELA 4s - Hispanic or Latino",
"Grade 6 Math 4s - All Students",
"Grade 6 Math 4s - Hispanic or Latino",
"Grade 6 Math 4s - Economically Disadvantaged",
"Grade 7 ELA 4s - All Students",
"Grade 8 ELA - All Students Tested",
"Grade 8 ELA 4s - All Students",
"Grade 8 ELA 4s - Hispanic or Latino",
"Grade 8 ELA 4s - Economically Disadvantaged",
"Grade 8 Math 4s - Economically Disadvantaged",                       
]]
School_Reg['Grade High'].value_counts()
School_Reg_test = School_Reg[School_Reg['Grade High'] == '08']
School_Reg_test.head()
School_Reg_test = School_Reg_test.dropna(axis=0)
X = School_Reg_test.iloc[:, [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]].values 
y_pred = RFC_best.predict(X)
y_pred
y_pred_df = pd.DataFrame({'Need':y_pred})
School_Reg_test = School_Reg_test.reset_index(drop=True)
Final_df =  pd.concat(objs=[School_Reg_test, y_pred_df], axis=1)
Final_df.head()
layout = go.Layout(title='Overall Stats of schools in need', width=500, height=500, margin=dict(l=50), xaxis=dict(tickangle=-35))
trace1 = go.Bar(x=Final_df['Need'].value_counts().index, y=Final_df['Need'].value_counts().values, marker=dict(color=Final_df['Need'].value_counts().values,colorscale = 'Rainbow'))

data = [trace1]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
import folium
from folium import plugins
from io import StringIO
import folium 

colors = ['red', 'yellow']
d = (Final_df['Need']).astype('int')
cols = [colors[int(i/1)] for i in d]

m = folium.Map([Final_df['Latitude'][0], Final_df['Longitude'][0]], zoom_start=10.3,tiles='stamentoner')

for lat, long, col in zip(Final_df['Latitude'], Final_df['Longitude'], cols):
    #rown = list(rown)
    folium.CircleMarker([lat, long], color=col, fill=True, radius=2).add_to(m)
m
v_features = Final_df.iloc[:, [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]].columns
plt.figure(figsize=(18,8*4))
gs = gridspec.GridSpec(9, 3)
for i, cn in enumerate(Final_df[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(Final_df[cn][Final_df.Need == 1], bins=50,color='blue')
    sns.distplot(Final_df[cn][Final_df.Need == 0], bins=50,color='red')
    ax.set_xlabel('')
    ax.set_title(str(cn))
plt.show()
Schoolarray = []
Feature1 = []
Schoolarray_good = []
Feature2 = []
#School Performance in Tests
Final_df["SPT"] = Final_df["Average ELA Proficiency"] + Final_df["Grade 8 ELA 4s - All Students"] + Final_df["Grade 5 ELA 4s - All Students"] + Final_df["Grade 6 ELA 4s - All Students"] + Final_df["Grade 6 Math 4s - All Students"] + Final_df["Grade 7 ELA 4s - All Students"]

#Black/Hispanic Students in Tests
Final_df["BHST"] = Final_df["Grade 8 ELA 4s - Hispanic or Latino"] + Final_df["Grade 6 Math 4s - Hispanic or Latino"] + Final_df["Grade 6 ELA 4s - Black or African American"] + Final_df["Grade 6 ELA 4s - Hispanic or Latino"]

#Effective School Management
Final_df["ESM"] = Final_df["Collaborative Teachers %"] + Final_df["Supportive Environment %"] + Final_df["Effective School Leadership %"] + Final_df["Trust %"]

#Performance of students who are Economically Disadvantaged
Final_df["EDS"] = Final_df["Grade 6 Math 4s - Economically Disadvantaged"] + Final_df["Grade 5 ELA 4s - Economically Disadvantaged"] + Final_df["Grade 8 ELA 4s - Economically Disadvantaged"] + Final_df["Grade 8 Math 4s - Economically Disadvantaged"]
Event_loc_df = Final_df[Final_df['Need'] == 0 ]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(Event_loc_df[['SPT']].values)
labels = kmeans.labels_

pct1 = Event_loc_df['SPT']
label1 = labels
label1 = np.array(label1)
pct1 = np.array(pct1)
idx = pct1.argsort()
label1, pct1 = [np.take(x, idx) for x in [label1, pct1]]

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(labels,Event_loc_df['SPT'],marker="o", color="Blue", s=10, linewidths=10)
ax.set_xlabel('Labels')
ax.set_ylabel('Overall School Performance in Tests')
ax.grid()
plt.grid()
print ('Schools with poor performance in Tests is with label "'+ str(label1[0])  + '" and I have used below map to show them.')
print ('Hover over the marked places to see the school names')
m = folium.Map([Final_df['Latitude'][0], Final_df['Longitude'][0]], zoom_start=10.3,tiles='stamentoner')
for (no, school, label, lat , long) in zip(Event_loc_df['SPT'], Event_loc_df['School Name'], labels ,Event_loc_df['Latitude'], Event_loc_df['Longitude']):
    if label == label1[0]:
       Schoolarray.append(school)
       Feature1.append('SPT')
       folium.CircleMarker([lat, long],
                    radius=5,
                    color='Red',
                    popup = school
                    ).add_to(m)
        
    if label == label1[len(label1) - 1]:
        Schoolarray_good.append(school)
        Feature2.append('SPT')
m     
Math_df = pd.read_csv("../input/new-york-state-math-test-results/2013-2015-new-york-state-mathematics-exam.csv")
Math_df.tail()
Math_df['Grade'].value_counts()
Math_df['Category'].value_counts()
plt.figure(figsize=(30,40))
g = sns.FacetGrid(data=Math_df,row='Category',col='Year')
g.map(sns.boxplot,'% Level 4')

Math_df_g8 = Math_df[Math_df['Grade'] == '8']
f,ax=plt.subplots(1,4,figsize=(25,12))
sns.boxplot(y='Category',x='% Level 1',data=Math_df_g8,ax=ax[0])
sns.boxplot(y='Category',x='% Level 2',data=Math_df_g8,ax=ax[1])
sns.boxplot(y='Category',x='% Level 3',data=Math_df_g8,ax=ax[2])
sns.boxplot(y='Category',x='% Level 4',data=Math_df_g8,ax=ax[3]);

Event_loc_df = Final_df[Final_df['Need'] == 0 ]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(Event_loc_df[['Percent Black / Hispanic']].values)
labels = kmeans.labels_

pct1 = Event_loc_df['Percent Black / Hispanic']
label1 = labels
label1 = np.array(label1)
pct1 = np.array(pct1)
idx = pct1.argsort()
label1, pct1 = [np.take(x, idx) for x in [label1, pct1]]

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(labels,Event_loc_df['Percent Black / Hispanic'],marker="o", color="Blue", s=10, linewidths=10)
ax.set_xlabel('Labels')
ax.set_ylabel('Percent Black / Hispanic')
ax.grid()
plt.grid()
print ('Schools with highest no of "Black/hispanic" is with label "'+ str(label1[len(label1) - 1])  + '" and i have used below map to show them')
print ('Hover over the marked places to see the school names')

m = folium.Map([Final_df['Latitude'][0], Final_df['Longitude'][0]], zoom_start=10.3,tiles='stamentoner')
for (no, school, label, lat , long) in zip(Event_loc_df['Percent Black / Hispanic'], Event_loc_df['School Name'], labels ,Event_loc_df['Latitude'], Event_loc_df['Longitude']):
    if label == label1[len(label1) - 1]:
       Schoolarray.append(school)
       Feature1.append('PBH')
       folium.CircleMarker([lat, long],
                    radius=5,
                    color='Red',
                    popup = school
                    ).add_to(m)
m     
Event_loc_df = Final_df[Final_df['Need'] == 0 ] 

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(Event_loc_df[['BHST']].values)
labels = kmeans.labels_

pct1 = Event_loc_df['BHST']
label1 = labels
label1 = np.array(label1)
pct1 = np.array(pct1)
idx = pct1.argsort()
label1, pct1 = [np.take(x, idx) for x in [label1, pct1]]


fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(labels,Event_loc_df['BHST'],marker="o", color="Blue", s=10, linewidths=10)
ax.set_xlabel('Labels')
ax.set_ylabel('Performance by Black/Hispanic students')
ax.grid()
plt.grid()
print ('Schools with poor performing Black/Hispanic students in tests are with label "'+ str(label1[0])  + '" and I have used below map to show them.')
print ('Hover over the marked places to see the school names')

m = folium.Map([Final_df['Latitude'][0], Final_df['Longitude'][0]], zoom_start=10.3,tiles='stamentoner')
for (no, school, label, lat , long) in zip(Event_loc_df['BHST'], Event_loc_df['School Name'], labels ,Event_loc_df['Latitude'], Event_loc_df['Longitude']):
    if label == label1[0]:
       Schoolarray.append(school)
       Feature1.append('BHST')
       folium.CircleMarker([lat, long],
                    radius=5,
                    color='Red',
                    popup = school
                    ).add_to(m)
        
    if label == label1[len(label1) - 1]:
        Schoolarray_good.append(school)
        Feature2.append('BHST')
m     

Event_loc_df = Final_df[Final_df['Need'] == 0 ] 

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(Event_loc_df[['EDS']].values)
labels = kmeans.labels_

pct1 = Event_loc_df['EDS']
label1 = labels
label1 = np.array(label1)
pct1 = np.array(pct1)
idx = pct1.argsort()
label1, pct1 = [np.take(x, idx) for x in [label1, pct1]]


fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(labels,Event_loc_df['EDS'],marker="o", color="Blue", s=10, linewidths=10)
ax.set_xlabel('Labels')
ax.set_ylabel('Poor performing students who are Economically disadvantaged')
ax.grid()
plt.grid()
print ('Schools having poor performing students who are Economically Disadvantaged are with label "'+ str(label1[0])  + '" and I have used below map to show them.')
print ('Hover over the marked places to see the school names')
m = folium.Map([Final_df['Latitude'][0], Final_df['Longitude'][0]], zoom_start=10.3,tiles='stamentoner')
for (no, school, label, lat , long) in zip(Event_loc_df['EDS'], Event_loc_df['School Name'], labels ,Event_loc_df['Latitude'], Event_loc_df['Longitude']):
    if label == label1[0]:
       Schoolarray.append(school)
       Feature1.append('EDS')
       folium.CircleMarker([lat, long],
                    radius=5,
                    color='Red',
                    popup = school
                    ).add_to(m)
        
    if label == label1[len(label1) - 1]:
        Schoolarray_good.append(school)
        Feature2.append('EDS')
m     
Event_loc_df = Final_df[Final_df['Need'] == 0 ] 

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(Event_loc_df[['ESM']].values)
labels = kmeans.labels_

pct1 = Event_loc_df['ESM']
label1 = labels
label1 = np.array(label1)
pct1 = np.array(pct1)
idx = pct1.argsort()
label1, pct1 = [np.take(x, idx) for x in [label1, pct1]]


fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(labels,Event_loc_df['ESM'],marker="o", color="Blue", s=10, linewidths=10)
ax.set_xlabel('Labels')
ax.set_ylabel('Poorly managed School')
ax.grid()
plt.grid()
print ('Poorly managed schools are with label "'+ str(label1[0])  + '" and i have used below map to show them')
print ('Hover over the marked places to see the school names')

m = folium.Map([Final_df['Latitude'][0], Final_df['Longitude'][0]], zoom_start=10.3,tiles='stamentoner')
for (no, school, label, lat , long) in zip(Event_loc_df['ESM'], Event_loc_df['School Name'], labels ,Event_loc_df['Latitude'], Event_loc_df['Longitude']):
    if label == label1[0]:
       Schoolarray.append(school)
       Feature1.append('ESM')
       folium.CircleMarker([lat, long],
                    radius=5,
                    color='Red',
                    popup = school
                    ).add_to(m)
m     
Lib_df = pd.read_csv("../input/nyc-queens-library-branches/queens-library-branches.csv")
Lib_df.head()
Lib_df.drop('notification', axis=1, inplace=True)
Lib_df = Lib_df.dropna(axis=0)
m = folium.Map(location=[40.75, -74],tiles='stamentoner')

for (_, (lat, long)) in Lib_df[['Latitude', 'Longitude']].iterrows():
    folium.CircleMarker([lat, long],
                    radius=5,
                    color='#3186cc',
                    fill_color='#3186cc',
                   ).add_to(m)
m
colors = ['red', 'yellow']
d = (Final_df['Need']).astype('int')
cols = [colors[int(i/1)] for i in d]

m = folium.Map([Final_df['Latitude'][0], Final_df['Longitude'][0]], zoom_start=10.3,tiles='stamentoner')

for lat, long, col in zip(Final_df['Latitude'], Final_df['Longitude'], cols):
    #rown = list(rown)
    folium.CircleMarker([lat, long], color=col, fill=True, radius=2).add_to(m)
    
for (_, (lat, long)) in Lib_df[['Latitude', 'Longitude']].iterrows():
    folium.CircleMarker([lat, long],
                    radius=5,
                    color='#3186cc',
                    fill_color='#3186cc',
                   ).add_to(m)
m
Event_loc_df = Final_df[Final_df['Need'] == 0 ]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8, random_state=0).fit(Event_loc_df[['Latitude', 'Longitude']].values)
labels = kmeans.labels_

colors = ['#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd']
m = folium.Map(location=[40.75, -74],tiles='stamentoner')

for (lat, long, label) in zip(Event_loc_df['Latitude'], Event_loc_df['Longitude'], labels):
    folium.CircleMarker([lat, long],
                    radius=5,
                    color=colors[label],
                    fill_color=colors[label],
                   ).add_to(m)

Event_label = pd.DataFrame({'labels':labels})
Event_loc_df = Event_loc_df.reset_index(drop=True)
Event_final =  pd.concat(objs=[Event_loc_df, Event_label], axis=1)
    
Event_lat = Event_final.groupby(['labels'])['Latitude'].mean()
Event_lon = Event_final.groupby(['labels'])['Longitude'].mean()
Event_lat = Event_lat.reset_index() 
Event_lon = Event_lon.reset_index() 

for (lat, long) in zip(Event_lat['Latitude'], Event_lon['Longitude']):
    folium.CircleMarker([lat, long],
                    radius=10,
                    color='red',
                    fill=True,
                   ).add_to(m)
m
Event_final1 = Event_final[['School Name','labels','Latitude','Longitude']]
Event_final1 = Event_final1.to_dict('records')
Event_lat1 = Event_lat.to_dict('records')
Event_lon1 = Event_lon.to_dict('records')

from math import cos, asin, sqrt
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(a))

def closest(data, v1,v2):
    return min(data, key=lambda p: distance(v1['Latitude'],v2['Longitude'],p['Latitude'],p['Longitude']))

v = {'lat': 40.7622290, 'lon': -73.77}
for i in range(8) :
   print(closest(Event_final1, Event_lat1[i],Event_lon1[i]))
eventid = [{'School Name': 'GOLDIE MAPLE ACADEMY', 'labels': 0, 'Latitude': 40.591349, 'Longitude': -73.78618900000001},
{'School Name': 'I.S. 313 SCHOOL OF LEADERSHIP DEVELOPMENT', 'labels': 1, 'Latitude': 40.840589, 'Longitude': -73.90454},
{'School Name': 'P.S. 178 SAINT CLAIR MCKELWAY', 'labels': 2, 'Latitude': 40.675234, 'Longitude': -73.915306},
{'School Name': 'P.S. 108 ASSEMBLYMAN ANGELO DEL TORO EDUCATIONAL COMPLEX', 'labels': 3, 'Latitude': 40.795035, 'Longitude': -73.947872},
{'School Name': 'P.S. 084 JOSE DE DIEGO', 'labels': 4, 'Latitude': 40.714822, 'Longitude': -73.963516},
{'School Name': 'P.S./M.S. 147 RONALD MCNAIR', 'labels': 5, 'Latitude': 40.698026, 'Longitude': -73.740151},
{'School Name': 'ICAHN CHARTER SCHOOL 3', 'labels': 6, 'Latitude': 40.856635, 'Longitude': -73.84304200000001},
{'School Name': 'STATEN ISLAND COMMUNITY CHARTER SCHOOL', 'labels': 7, 'Latitude': 40.630238, 'Longitude': -74.08114300000001}]
m = folium.Map(location=[40.75, -74],tiles='stamentoner')
for (lat, long, label) in zip(Event_loc_df['Latitude'], Event_loc_df['Longitude'], labels):
    folium.CircleMarker([lat, long],
                    radius=5,
                    color=colors[label],
                    fill_color=colors[label],
                   ).add_to(m)

for i in range(8) :
    folium.Marker([eventid[i]['Latitude'],eventid[i]['Longitude']],
                    popup=eventid[i]['School Name'],
                    icon=folium.Icon(color='Red')
                    ).add_to(m)

m    
Event_loc_df = Final_df[Final_df['Need'] == 0 ] 

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(Event_loc_df[['Percent of Students Chronically Absent']].values)
labels = kmeans.labels_

pct1 = Event_loc_df['Percent of Students Chronically Absent']
label1 = labels
label1 = np.array(label1)
pct1 = np.array(pct1)
idx = pct1.argsort()
label1, pct1 = [np.take(x, idx) for x in [label1, pct1]]


fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(labels,Event_loc_df['Percent of Students Chronically Absent'],marker="o", color="Blue", s=10, linewidths=10)
ax.set_xlabel('Labels')
ax.set_ylabel('Percent of Students Chronically Absent')
ax.grid()
plt.grid()
print ('Schools with highest no of absent percentage is with label "'+ str(label1[len(label1) - 1])  + '" and i have used below map to show them')
print ('Hover over the marked places to see the school names')

m = folium.Map([Final_df['Latitude'][0], Final_df['Longitude'][0]], zoom_start=10.3,tiles='stamentoner')
for (no, school, label, lat , long) in zip(Event_loc_df['Percent of Students Chronically Absent'], Event_loc_df['School Name'], labels ,Event_loc_df['Latitude'], Event_loc_df['Longitude']):
    if label == label1[len(label1) - 1]:
       Schoolarray.append(school)
       Feature1.append('PSCA')
       folium.CircleMarker([lat, long],
                    radius=5,
                    color='Red',
                    popup = school
                    ).add_to(m)
m     

Safety_df = pd.read_csv("../input/ny-2010-2016-school-safety-report/2010-2016-school-safety-report.csv")
Safety_df.tail()
Safety_df_filter = Safety_df[['School Year','Latitude', 'Longitude','Major N','Oth N','NoCrim N','Prop N','Vio N']]
Safety_df_filter = Safety_df_filter[Safety_df_filter['School Year'] == '2015-16']

Safety_df1 = Safety_df_filter[Safety_df_filter['Major N'] > 1.0 ]
Safety_df2 = Safety_df_filter[Safety_df_filter['Oth N'] > 1.0 ]
Safety_df3 = Safety_df_filter[Safety_df_filter['NoCrim N'] > 1.0 ]
Safety_df4 = Safety_df_filter[Safety_df_filter['Prop N'] > 1.0 ]
Safety_df5 = Safety_df_filter[Safety_df_filter['Vio N'] > 1.0 ]

Safety_df1 = Safety_df1.dropna(axis=0)
Safety_df2 = Safety_df2.dropna(axis=0)
Safety_df3 = Safety_df3.dropna(axis=0)
Safety_df4 = Safety_df4.dropna(axis=0)
Safety_df5 = Safety_df5.dropna(axis=0)

count1 = Safety_df1['Major N'].values
count2 = Safety_df2['Oth N'].values
count3 = Safety_df3['NoCrim N'].values
count4 = Safety_df4['Prop N'].values
count5 = Safety_df5['Vio N'].values
m = folium.Map(location=[40.75, -74],tiles='stamentoner')
for i in range(0,Safety_df1['School Year'].count()):
   folium.Circle(
      location=[Safety_df1.iloc[i]['Latitude'], Safety_df1.iloc[i]['Longitude']],
      #location=[20, 81],
      radius=int(count1[i])*100,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)

for i in range(0,Safety_df2['School Year'].count()):
   folium.Circle(
      location=[Safety_df2.iloc[i]['Latitude'], Safety_df2.iloc[i]['Longitude']],
      #location=[20, 81],
      radius=int(count2[i])*100,
      color='yellow',
      fill=True,
      fill_color='yellow'
   ).add_to(m)

for i in range(0,Safety_df3['School Year'].count()):
   folium.Circle(
      location=[Safety_df3.iloc[i]['Latitude'], Safety_df3.iloc[i]['Longitude']],
      #location=[20, 81],
      radius=int(count3[i])*100,
      color='purple',
      fill=True,
      fill_color='purple'
   ).add_to(m)

for i in range(0,Safety_df4['School Year'].count()):
   folium.Circle(
      location=[Safety_df4.iloc[i]['Latitude'], Safety_df4.iloc[i]['Longitude']],
      #location=[20, 81],
      radius=int(count4[i])*100,
      color='green',
      fill=True,
      fill_color='green'
   ).add_to(m)

for i in range(0,Safety_df5['School Year'].count()):
   folium.Circle(
      location=[Safety_df5.iloc[i]['Latitude'], Safety_df5.iloc[i]['Longitude']],
      #location=[20, 81],
      radius=int(count5[i])*100,
      color='blue',
      fill=True,
      fill_color='blue'
   ).add_to(m)

m
Safety_df['Major N'].fillna(0, inplace=True)
Safety_df['Oth N'].fillna(0, inplace=True)
Safety_df['NoCrim N'].fillna(0, inplace=True)
Safety_df['Prop N'].fillna(0, inplace=True)
Safety_df['Vio N'].fillna(0, inplace=True)

Safety_df['Risk_idx'] = (2*Safety_df['Major N']) + (1.5*Safety_df['Vio N']) + Safety_df['Prop N'] + (0.5*Safety_df['Oth N'])+(0.1*Safety_df['NoCrim N'])

Safety_raw = Safety_df.groupby(['Location Name', 'School Year'])['Risk_idx'].sum().unstack()
Safety_raw.columns.name = None      
Safety_raw = Safety_raw.reset_index()  
Safety_raw = Safety_raw[['Location Name','2013-14','2014-15','2015-16']]
Safety_raw['2013-14'].fillna(0, inplace=True)
Safety_raw['2014-15'].fillna(0, inplace=True)
Safety_raw['2015-16'].fillna(0, inplace=True)
Feature = Safety_raw['Location Name']
Y14 = Safety_raw['2013-14']
Y15 = Safety_raw['2014-15']
Y16 = Safety_raw['2015-16']

Feature = np.array(Feature)
Y14 = np.array(Y14)
Y15 = np.array(Y15)
Y16 = np.array(Y16)

idx = Y16.argsort()
Feature, Y14, Y15, Y16 = [np.take(x, idx) for x in [Feature, Y14, Y15, Y16]]

Feature = Feature[2260:]
Y14 = Y14[2260:]
Y15 = Y15[2260:]
Y16 = Y16[2260:]

s = 1
size=[]
for i, cn in enumerate(Y16):
     s = s + 1        
     size.append(s)

fig, ax = plt.subplots(figsize=(13, 8))
ax.scatter(Y16,size,marker="o", color="lightBlue", s=size, linewidths=10)
ax.set_xlabel('Risk Index')
ax.set_ylabel('')
ax.set_title('School locations with high Risk index Year 15-16')
#ax.spines['right'].set_visible(True)
ax.grid()
for i, txt in enumerate(Feature):
      ax.annotate(txt, (Y16[i],size[i]),fontsize=12,rotation=0,color='Green')

Safety_school_merged = pd.merge(Safety_df, Safety_raw, how='left', left_on='Location Name', right_on='Location Name')
Safety_school_merged = Safety_school_merged.drop_duplicates(subset=['Location Name'])
Risky_areas = Safety_school_merged[Safety_school_merged['2015-16'] > 30]

m = folium.Map(location=[40.75, -74],tiles='stamentoner')
for (lat, long, label) in zip(Event_loc_df['Latitude'], Event_loc_df['Longitude'], labels):
    folium.CircleMarker([lat, long],
                    radius=5,
                    color=colors[label],
                    fill_color=colors[label],
                   ).add_to(m)

for (lat, long,name) in zip(Risky_areas['Latitude'], Risky_areas['Longitude'],Risky_areas['Location Name']):
    folium.Marker([lat, long],
                   icon=folium.Icon(color='Red'),
                   popup= name,
                    ).add_to(m)

m    
medicaid_df = pd.read_csv("../input/nyc-medical-assistance-program-medicaid-offices/medical-assistance-program-medicaid-offices.csv")
medicaid_df
m = folium.Map(location=[40.75, -74],tiles='stamentoner')
for (lat, long, label) in zip(Event_loc_df['Latitude'], Event_loc_df['Longitude'], labels):
    folium.CircleMarker([lat, long],
                    radius=5,
                    color=colors[label],
                    fill_color=colors[label],
                   ).add_to(m)

for (lat, long,name) in zip(medicaid_df['Latitude'], medicaid_df['Longitude'],medicaid_df['Name of Medicaid Offices']):
    folium.Marker([lat, long],
                   icon=folium.Icon(color='Red'),
                   popup= name,
                    ).add_to(m)

m    
SchoolArray_df = pd.DataFrame({'Schools':Schoolarray,
                              'Features':Feature1})
SchoolArray_df1 = SchoolArray_df.groupby(['Schools','Features'])['Features'].count().unstack()
SchoolArray_df1.columns.name = None      
SchoolArray_df1 = SchoolArray_df1.reset_index()  
SchoolArray_df1 = SchoolArray_df1[['Schools','BHST','EDS','SPT','ESM','PBH','PSCA']]
SchoolArray_df1 = SchoolArray_df1.fillna(0)
SchoolArray_df1['Occurance'] = SchoolArray_df1['BHST'] + SchoolArray_df1['EDS'] + SchoolArray_df1['ESM'] + SchoolArray_df1['PBH']+ SchoolArray_df1['PSCA'] + SchoolArray_df1['SPT']
SchoolArray_df1 = SchoolArray_df1.sort_values(['Occurance'], ascending=False)
#f, ax = plt.subplots(figsize=(15,50)) 
#g = sns.heatmap(SchoolArray_df1,annot=True,cmap='YlGnBu',linewidths=.5,vmin=0.01)
#plt.show()
cm = sns.light_palette("Yellow", as_cmap=True)
SchoolArray_df1.style.background_gradient(cmap=cm)
index = SchoolArray_df['Schools'].value_counts().index
value = SchoolArray_df['Schools'].value_counts().values
School_ranking = pd.DataFrame({'Schools':index,
                          'Occurrence':value})
School_ranking_loc = pd.merge(School_ranking, School_df, how='left', left_on='Schools', right_on='School Name')
School_ranking[School_ranking['Occurrence']== 6]
School_ranking_loc1 = School_ranking_loc[School_ranking_loc['Occurrence']>5]
m = folium.Map(location=[40.75, -74],tiles='stamentoner')
for (lat, long,name) in zip(School_ranking_loc1['Latitude'], School_ranking_loc1['Longitude'],School_ranking_loc1['Schools']):
    folium.Marker([lat, long],
                   icon=folium.Icon(color='Red'),
                   popup= name,
                    ).add_to(m)

m
SchoolArray_good_df = pd.DataFrame({'Schools':Schoolarray_good,
                                  'Features':Feature2})
SchoolArray_df2 = SchoolArray_good_df.groupby(['Schools','Features'])['Features'].count().unstack()
SchoolArray_df2.columns.name = None      
SchoolArray_df2 = SchoolArray_df2.reset_index()  
SchoolArray_df2 = SchoolArray_df2[['Schools','BHST','EDS','SPT']]
SchoolArray_df2 = SchoolArray_df2.fillna(0)
SchoolArray_df2['Occurance'] = SchoolArray_df2['BHST'] + SchoolArray_df2['EDS'] + SchoolArray_df2['SPT']
SchoolArray_df2 = SchoolArray_df2.sort_values(['Occurance'], ascending=False)
#f, ax = plt.subplots(figsize=(15,30)) 
#g = sns.heatmap(SchoolArray_df2,annot=True,cmap='YlGnBu',linewidths=.5,vmin=0.01)
#plt.show()
cm = sns.light_palette("Yellow", as_cmap=True)
SchoolArray_df2.style.background_gradient(cmap=cm)
SchoolArray_good_df = pd.DataFrame(Schoolarray_good,columns=['schools'])
index = SchoolArray_good_df['schools'].value_counts().index
value = SchoolArray_good_df['schools'].value_counts().values
good_School_ranking = pd.DataFrame({'schools':index,
                          'occurrence':value})
good_School_ranking_loc = pd.merge(good_School_ranking, School_df, how='left', left_on='schools', right_on='School Name')

School_ranking_loc2 = good_School_ranking_loc[good_School_ranking_loc['occurrence']>2]
m = folium.Map(location=[40.75, -74],tiles='stamentoner')
for (lat, long,name) in zip(School_ranking_loc2['Latitude'], School_ranking_loc2['Longitude'],School_ranking_loc2['schools']):
    folium.Marker([lat, long],
                   icon=folium.Icon(color='Red'),
                   popup= name,
                    ).add_to(m)

m
PoortoAverage_schools = pd.merge(SchoolArray_df1, School_df, how='left', left_on='Schools', right_on='School Name')
GoodtoAverage_schools = pd.merge(SchoolArray_df2, School_df, how='left', left_on='Schools', right_on='School Name')
PoortoAverage_schools[(PoortoAverage_schools['PBH']== 1) & (PoortoAverage_schools['BHST']== 1)].tail(5)
PoortoAverage_schools[(PoortoAverage_schools['SPT']== 0) & (PoortoAverage_schools['ESM']== 1)].tail(5)
GoodtoAverage_schools[(GoodtoAverage_schools['EDS']== 1)].tail(5)
PoortoAverage_schools.to_csv("PoortoAverage_list.csv")
GoodtoAverage_schools.to_csv("GoodtoAverage_list.csv")