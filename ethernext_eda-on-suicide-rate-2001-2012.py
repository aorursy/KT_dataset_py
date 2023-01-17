import numpy as np

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





#Plotly libraries

import plotly as py

import plotly.graph_objs as go

import plotly.express as px

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
data=pd.read_csv('../input/suicides-in-india/Suicides in India 2001-2012.csv')

data.info()
data.sample(10)
# rename states



data.replace('A & N Islands (Ut)', 'A & N Islands', inplace=True)

data.replace('Chandigarh (Ut)', 'Chandigarh', inplace=True)

data.replace('D & N Haveli (Ut)', 'D & N Haveli', inplace=True)

data.replace('Daman & Diu (Ut)', 'Daman & Diu', inplace=True)

data.replace('Lakshadweep (Ut)', 'Lakshadweep', inplace=True)

data.replace('Delhi (Ut)', 'Delhi', inplace=True)

# rename Type



data.replace('Bankruptcy or Sudden change in Economic', 

           'Bankruptcy or Sudden change in Economic Status', inplace=True)

data.replace('By Other means (please specify)', 'By Other means', inplace=True)

data.replace('Not having Children(Barrenness/Impotency',

           'Not having Children (Barrenness/Impotency', inplace=True)
data = data.drop(data[(data.State == 'Total (Uts)') | (data.State == 'Total (All India)') | 

               (data.State == 'Total (States)')].index)

data=data.drop(data[(data.Type =='By Other means')|(data.Type=='Other Causes (Please Specity)')|

                    (data.Type=='Others (Please Specify)')|(data.Type=='Causes Not known')].index)
data = data.drop(data[data.Total==0].index)
fig = px.bar(data, x="State", y="Total", color="State",

  animation_frame="Year", animation_group="Total", range_y=[0,20000],width=1000)

py.offline.iplot(fig)
temp_state = data.groupby('State').count()['Total'].reset_index().sort_values(by='Total',ascending=False)

temp_state.style.background_gradient(cmap='Reds')
counts = data['Gender'].value_counts().sort_index()

print(counts)

# Plot a pie chart

counts.plot(kind='pie', title='Gender Count',figsize=(10,8))



plt.legend()

plt.show()
# splitting data as per the type code



cause = data[data['Type_code'] == 'Causes']

edu_status = data[data['Type_code'] == 'Education_Status']

means_adpt = data[data['Type_code'] == 'Means_adopted']

prof = data[data['Type_code'] == 'Professional_Profile']

soc_status = data[data['Type_code'] == 'Social_Status']
def plot_type(data, Title, X_lab):

    p_type = data.groupby('Type').sum()['Total']

    sort_df = p_type.sort_values(ascending = False)



    fig = sort_df.plot(kind='bar', figsize = (10,6), title = Title + '\n', width = 0.75)

    fig.set_xlabel('\n' + X_lab )

    fig.set_ylabel('Count\n')

    sns.set_style('whitegrid')

    sns.set_palette('Set2')  
# plot by Cause

plot_type(cause, 'Suicide by cause', 'Cause')
#plot by the educational causes

plot_type(edu_status, 'Suicide by Education Status', 'Education Status')
# plot by means adopted

plot_type(means_adpt, 'Suicide by Means Adopted', 'Means Adopted')
# suicide by professional profile

plot_type(prof, 'Suicide by Professional Profile', 'Professional Profile')
# suicide by social Status

plot_type(soc_status, 'Suicide by Social Status', 'Social Status')
#Splitting the data as per the State

State1 = data[data['State']=='Karnataka']

State2 = data[data['State']=='Tamil Nadu']

State3 = data[data['State']=='Andhra Pradesh']

def plot_for_State_by_age(data):

    plt.figure(figsize=(12,6))

    data = data[['Age_group','Gender','Total']]

    edSort = data.groupby(['Age_group','Gender'],as_index=False).sum().sort_values('Total',ascending=False)

    sns.barplot(x='Age_group',y='Total',hue='Gender',data=edSort,palette='RdBu')
def plot_for_State_by_type(data):

    plt.figure(figsize=(12,6))

    data = data[['Type_code','Gender','Total']]

    edSort = data.groupby(['Type_code','Gender'],as_index=False).sum().sort_values('Total',ascending=False)

    sns.barplot(x='Type_code',y='Total',hue='Gender',data=edSort,palette='ch:2.5,-.2,dark=.3')
#plotting as per the age_group

plot_for_State_by_age(State1)
#plotting as per the differnet causes

plot_for_State_by_type(State1)
#plotting as per the age_group

plot_for_State_by_age(State2)
#plotting as per the differnet causes

plot_for_State_by_type(State2)
#plotting as per the age_group

plot_for_State_by_age(State3)
#plotting as per the differnet causes

plot_for_State_by_type(State3)