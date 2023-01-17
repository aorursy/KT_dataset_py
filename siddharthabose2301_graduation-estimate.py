import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

from plotly.subplots import make_subplots

import plotly.graph_objects as go



%matplotlib inline
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
print("The data frame has {} rows and {} columns".format(df.shape[0],df.shape[1]))
df.head()
df.tail()
df.drop('sl_no',axis = 1,inplace = True)
df.info()
#getting the right data types to reduce memory usage



df['gender'] = df['gender'].astype('category')

df['ssc_b'] = df['ssc_b'].astype('category')

df['hsc_b'] = df['hsc_b'].astype('category')

df['hsc_s'] = df['hsc_s'].astype('category')

df['degree_t'] = df['degree_t'].astype('category')

df['workex'] = df['workex'].astype('category')

df['specialisation'] = df['specialisation'].astype('category')

df['status'] = df['status'].astype('category')
df.info()
df[df['status']=='Not Placed'][['status','salary']].info()
df[df['status']=='Placed'][['status','salary']].info()
#removing the nan

df['salary'].fillna(0,inplace = True)
df.info()
data_gen = df.groupby(['gender','status']).count()['salary'].reset_index()

data_gen.columns = ['gender','status','count']

fig = px.bar(data_gen,x = 'gender',y = 'count',color = 'status',barmode = 'group')

fig.update_layout(width = 800,title  = 'Gender wise Placements')

fig.show()
fig = px.box(df[df['status']=='Placed'],x = 'gender',y = 'salary',color = 'gender',width = 500,title = 'Gender vs Salary')

fig.show()
fig = px.box(df[df['status']=='Placed'],x = 'ssc_b',y = 'salary',color = 'ssc_b',width = 500,title = 'SSC Board vs Salary')

fig.show()
fig = px.box(df[df['status']=='Placed'],x = 'hsc_s',y = 'salary',color = 'hsc_s',width = 500,title = 'SSC Board vs Salary')

fig.show()
data_gen = df.groupby(['hsc_s','status']).count()['salary'].reset_index()

data_gen.columns = ['hsc_s','status','count']

fig = px.bar(data_gen,x = 'hsc_s',y = 'count',color = 'status',barmode = 'group')

fig.update_layout(width = 800,title  = 'Stream wise Placements')

fig.show()
data_gen = df.groupby(['workex','status']).count()['salary'].reset_index()

data_gen.columns = ['workex','status','count']

fig = px.bar(data_gen,x = 'workex',y = 'count',color = 'status',barmode = 'group')

fig.update_layout(width = 800,title  = 'Work Expirence dependency on Placements')

fig.show()
fig = px.box(df[df['status']=='Placed'],x = 'specialisation',y = 'salary',color = 'specialisation',width = 500,title = 'Specialisation vs Salary')

fig.show()