import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

import plotly.express as px

import plotly.graph_objects as go

data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
import seaborn as sns

sns.countplot(data.workex)
sns.countplot(data.gender)
fig = go.Figure([go.Pie(labels=data['status'].unique(), values=data['status'].value_counts())])

fig.show()
fig = go.Figure([go.Pie(labels=data['degree_t'].unique(), values=data['degree_t'].value_counts())])

fig.show()
data.groupby('gender')['status'].value_counts()
plt.figure(figsize=(10,7))

sns.boxplot(x='status',y='ssc_p',data=data)
plt.figure(figsize=(10,7))

sns.boxplot(x='status',y='hsc_p',data=data)
sns.distplot(data['ssc_p'])
sns.distplot(data['hsc_p'])
sns.distplot(data['degree_p'])
commerce = data[data['hsc_s']=='Commerce']['hsc_p']

science = data[data['hsc_s']=='Science']['hsc_p']

arts = data[data['hsc_s']=='Arts']['hsc_p']



plt.figure(figsize=(10,5))

sns.distplot(commerce, color="skyblue", label="Commerce",hist=False)

sns.distplot(science, color="red", label="Science",hist=False)

sns.distplot(arts, color="green", label="Arts",hist=False)
tech = data[data['degree_t']=='Sci&Tech']['degree_p']

com_mgt = data[data['degree_t']=='Comm&Mgmt']['degree_p']

others = data[data['degree_t']=='Others']['degree_p']



plt.figure(figsize=(10,5))

sns.distplot(tech, color="skyblue", label="Science & Tech",hist=False)

sns.distplot(com_mgt, color="red", label="Commerce & Management",hist=False)

sns.distplot(others, color="green", label="Others",hist=False)
plt.figure(figsize=(12,5))

sns.scatterplot(y='degree_p',x='salary',data=data)
plt.figure(figsize=(12,5))

sns.scatterplot(y='mba_p',x='salary',data=data)
plt.figure(figsize=(12,5))

sns.scatterplot(x='etest_p',y='degree_p',hue='workex',data=data)
plt.figure(figsize=(12,5))

sns.scatterplot(y='etest_p',x='salary',data=data)
plt.figure(figsize=(12,5))

sns.scatterplot(x='salary',y='degree_p',hue='workex',data=data)
plt.figure(figsize=(12,5))

sns.scatterplot(x='salary',y='mba_p',hue='specialisation',data=data)