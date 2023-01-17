# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
! pip install dexplot
# Importing the necessary Libraries

# Importing the necessary libraries

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objs as go
from plotly.offline import iplot
df = pd.read_csv('/kaggle/input/trial-and-terror/trial_and_terror.csv')
df.head()
# Getting info about the datatype

df.info()
# Shape of the data
df.shape
# Describing the data
df.describe()
# Cheking for the null values in the dataset

df.isnull().sum()
# For finer-grained control, the thresh parameter lets you specify a minimum number of non-null values for the row/column to be kept
# Here I am taking that threshold as 408, so any row having values greater than or equal to 408 will be kept intact

df.dropna(axis='columns', thresh=408, inplace = True)
df
df['description']
labels = df['race'].value_counts().index
values = df['race'].value_counts().values

colors = df['race']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", marker=dict(colors=colors))])
fig.show()
# Lets try to plot in how many cases guilty person has been released

labels = df['released'].value_counts().index
values = df['released'].value_counts().values

colors = df['released']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", marker=dict(colors=colors))])
fig.show()
# Lets see the disposition status of the guilty person

labels = df['disposition'].value_counts().index
values = df['disposition'].value_counts().values

colors = df['disposition']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", marker=dict(colors=colors))])
fig.show()
# Lets look at the gender 

labels = df['gender'].value_counts().index
values = df['gender'].value_counts().values

colors = df['gender']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", marker=dict(colors=colors))])
fig.show()
# Plotting the instituitional name display

labels = df['institution_nameDisplay'].value_counts().index
values = df['institution_nameDisplay'].value_counts().values

colors = df['institution_nameDisplay']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", marker=dict(colors=colors))])
fig.show()
# Plotting the instituition city where crime has been done

labels = df['institution_city'].value_counts().index
values = df['institution_city'].value_counts().values

colors = df['institution_city']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", marker=dict(colors=colors))])
fig.show()
# Plotting the instituition state where crime has been done

labels = df['institution_state'].value_counts().index
values = df['institution_state'].value_counts().values

colors = df['institution_state']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", marker=dict(colors=colors))])
fig.show()


# Plotting the institution_securityLevel where crime has been done

labels = df['institution_securityLevel'].value_counts().index
values = df['institution_securityLevel'].value_counts().values

colors = df['institution_securityLevel']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", marker=dict(colors=colors))])
fig.show()
# Lets plot the data of case_district for top 10 districts having the max number of the cases

labels = df['case_district'].value_counts()[:10].index
values = df['case_district'].value_counts()[:10].values

colors = df['case_district']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", marker=dict(colors=colors))])
fig.show()
# Lets plot the data of case_imprisonment
labels = df['case_imprisonment'].value_counts().index
values = df['case_imprisonment'].value_counts().values

colors = df['case_imprisonment']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", marker=dict(colors=colors))])
fig.show()
# Lets plot the data of case_restitutiont 
labels = df['case_state'].value_counts().index
values = df['case_state'].value_counts().values

colors = df['case_state']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", marker=dict(colors=colors))])
fig.show()

# Lets plot the data of case_restitutiont 

labels = df['case_sting'].value_counts().index
values = df['case_sting'].value_counts().values

colors = df['case_sting']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", marker=dict(colors=colors))])
fig.show()
# Lets plot the data of first case charge of the criminals 
labels = df['case_charge_1'].value_counts().index
values = df['case_charge_1'].value_counts().values

colors = df['case_charge_1']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", marker=dict(colors=colors))])
fig.show()
# Lets plot the data of second case charge of the criminals 

labels = df['case_charge_2'].value_counts().index
values = df['case_charge_2'].value_counts().values

colors = df['case_charge_2']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", marker=dict(colors=colors))])
fig.show()
  

# Lets plot the data of terror org of the criminals 

labels = df['case_terrorOrg_1'].value_counts().index
values = df['case_terrorOrg_1'].value_counts().values

colors = df['case_terrorOrg_1']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent", marker=dict(colors=colors))])
fig.show()
import dexplot as dxp

dxp.count(val='race', data=df, figsize=(4,3), normalize=True)
# Lets plot the relation between the race and the released data of the criminals

dxp.count(val='released', data=df, split='race', figsize=(10,7), normalize=True)
# Lets plot the relation between the race and the disposition status of the criminals

dxp.count(val='race', data=df, split='disposition', figsize=(10,7), normalize=True)
# Lets plot the relation between the race and the gender status of the criminals

dxp.count(val='race', data=df, split='gender', figsize=(10,7), normalize=True)
# Lets plot the relation between the released and the gender status of the criminals

dxp.count(val='released', data=df, split='gender', figsize=(10,7), normalize=True)
# Lets plot the relation between the released and the disposition status of the criminals

dxp.count(val='released', data=df, split='disposition', figsize=(10,7), normalize=True)
# Lets plot the relation between the disposition and the gender status of the criminals

dxp.count(val='disposition', data=df, split='gender', figsize=(10,7), normalize=True)
# Visualize the relation between the race and the terror organization they are asscociated with

dxp.count(
val='case_terrorOrg_1',
data=df,
split='race',
orientation="h",
stacked=True,
figsize=(6,4),
xlabel="Race Distribution")

# Visualize the relation between the race and the case charge 1 they are asscociated with

dxp.count(
val='case_charge_1',
data=df,
split='race',
orientation="h",
stacked=True,
figsize=(8,6),
xlabel="Race Distribution")

# Visualize the relation between the race and the case charge 2 they are asscociated with

dxp.count(
val='case_charge_2',
data=df,
split='race',
orientation="h",
stacked=True,
figsize=(8,6),
xlabel="Race Distribution")
# Visualize the relation between the race and the state in which the crime has been committed

dxp.count(
val='case_state',
data=df,
split='race',
orientation="h",
stacked=True,
figsize=(8,6),
xlabel="Race Distribution")


# Visualize the relation between the race and the case_imprisonment of the crime  committed

dxp.count(
val='case_imprisonment',
data=df,
split='race',
orientation="h",
stacked=True,
figsize=(12,8),
xlabel="Race Distribution")
# Visualize the relation between the race and the case_imprisonment of the crime  committed

dxp.count(
val='case_district',
data=df,
split='race',
orientation="h",
stacked=True,
figsize=(12,8),
xlabel="Race Distribution")