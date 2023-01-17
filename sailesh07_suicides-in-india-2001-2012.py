import ipywidgets as widgets

from IPython.display import display

from ipywidgets import widgets, interactive

from ipywidgets import interact

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import pandas_profiling

import statsmodels

df=pd.read_csv("../input/Suicides in India 2001-2012.csv")

df.head()
df.info()
df.info(memory_usage='deep')
df.memory_usage(deep=True)
df.memory_usage(deep=True).sum()
df['State'] = df['State'].astype('category')

df['Type_code'] = df['Type_code'].astype('category')

df['Gender'] = df['Gender'].astype('category')

df['Type'] = df['Type'].astype('category')

df['Age_group'] = df['Age_group'].astype('category')
df.dtypes
df['State'].head()
df.State.cat.codes.head()
df.memory_usage(deep = True)
df.info('deep')
df.memory_usage(deep = True).sum()
pandas_profiling.ProfileReport(df)
df.shape
df.describe()
df.info()
df.isna().sum()
df_copy = df.copy()
df.plot()
df['Total'].plot()
df['Total'].max()
df[df["Total"]==63343]
df["State"].unique()
df_all_ind=df[df['State']=="Total (All India)"]

df_all_ind.head()
df_all_ind["Total"].sum()
df['Type_code'].unique()
df['Type'].unique()
df['Type'].nunique()
df.head()
df =df[df['Total']>0]
df.head()
df.shape
df['Type_code'].value_counts()
plt.title('Type Code')

df['Type_code'].value_counts().plot(kind='bar')
df['Type'].value_counts(normalize=True)
plt.title('Type of suicides')

df['Type'].value_counts(normalize=False).plot(kind='bar',figsize=(20,12))
df['Age_group'].value_counts()
plt.title('Age group')

df['Age_group'].value_counts().plot(kind='bar')
df['Gender'].value_counts(normalize=True)
plt.title('Male and Female')

df['Gender'].value_counts().plot(kind='bar')
df.head()
df_year_total = df.groupby('Year')['Total'].sum()
df_year_total = pd.DataFrame(df_year_total)
df_year_total.reset_index(inplace=True)
df_year_total.head()
fig =px.bar(df_year_total,x='Year',y='Total',labels={'Total':'No of suicides: '}, height=400,color='Year')

fig.update_layout(title_text='Total no of suicides by year from 2001 - 2012')

fig.show()
 
df_grnder = df.groupby('Gender')['Total'].sum()

df_grnder=pd.DataFrame(df_grnder)

df_grnder.reset_index(inplace=True)

df_grnder.head()
fig=px.bar(df_grnder,x='Gender',y='Total',labels={'Total':'No of suicides: '},height=400,color='Gender')

fig.update_layout(title_text='Suicides of Male and Female from 2001 - 2012')

fig.show()
df_type=df.groupby(['Type','Gender'])['Total'].sum()

df_type=pd.DataFrame(df_type)

df_type.reset_index(inplace=True)

df_type.sort_values(by="Total" , inplace=True ,ascending=False)



df_type.head()
fig=px.bar(df_type,x='Type',y='Total',labels={'Total':'No of suicides: '},height=1000,width=1800,color='Gender',barmode='group')

fig.update_layout(title_text = 'List of types and total no of suicides')

fig.show()
fig=px.bar(df_type[:7],x='Type',y='Total',labels={'Total':'No of suicides: '},height=600,width=1500,color='Gender',barmode='group')

fig.update_layout(title_text = 'List of top 7 types of suicides')

fig.show()
df_state = df.groupby('State')['Total'].sum()

df_state = pd.DataFrame(df_state)

df_state.reset_index(inplace= True)

df_state.sort_values(by='Total',inplace=True,ascending = False)

df_state.head()
df_state.drop(df_state.index[[0,1]],inplace=True)
df_state.head()
fig = px.bar(df_state,x='State',y='Total',labels={'Total': 'No of Suicides: '}, height=400,color='Total')

fig.update_layout(title_text = 'List of states and their total no of suicides ')

fig.show()
fig = px.bar(df_state[:7],x='State',y='Total',labels={'Total': 'No of Suicides: '}, height=400,color='Total')

fig.update_layout(title_text = 'List of Top 7 States and their total no of suicides')

fig.show()
df_state.sort_values(by='Total',inplace=True,ascending=True)

df_state.head()
fig =px.bar(df_state[:7],x='State',y='Total',labels={'Total': 'No of Suicides: '}, height=400,color='Total')

fig.update_layout(title_text = 'States with lowest number of suicides from 2001-2012')

fig.show()
df.head()
df_state_year = df.groupby(['State','Year','Gender'])['Total'].sum()

df_state_year = pd.DataFrame(df_state_year)

df_state_year.reset_index(inplace = True)

df_state_year.head(5)
df_Island_Andhra=df_state_year[df_state_year.State.isin(['A & N Islands','Andhra Pradesh'])]

df_Island_Andhra.head()
fig=px.bar(df_Island_Andhra,x='Year',y='Total',facet_col='State',color='State')

fig.update_layout(title_text = 'Suicides in Andhra And A & N island over years')

fig.show()
df.head()
df_state_and_type=df.groupby(['State','Type'])['Total'].sum()

df_state_and_type = pd.DataFrame(df_state_and_type)

df_state_and_type.sort_values(by='Total',inplace=True,ascending =False)

df_state_and_type.reset_index(inplace=True)

df_state_and_type
df_state_and_type['State'].unique()
df_Kl_TN = df_state_and_type[df_state_and_type.State.isin(['Kerala','Tamil Nadu'])]

df_Kl_TN
df_Kl_TN.Type.unique()
df_Kl_TN=df_Kl_TN[df_Kl_TN['Total']>500]
df_Kl_TN
df_Kl_TN = df_Kl_TN[df_Kl_TN['Total']>20000]

df_Kl_TN.head()
plt.figure(figsize=(20,10))

g=sns.barplot('Type',y='Total',data=df_Kl_TN)

g.set_xticklabels(g.get_xticklabels(), rotation=90,horizontalalignment='right')

plt.show()
type(g)
df.head()
df_Type_suicide = df.groupby(['Year','Type_code','Type','Gender'])['Total'].sum()

df_Type_suicide = pd.DataFrame(df_Type_suicide)

df_Type_suicide.reset_index(inplace=True)

df_Type_suicide.head()
fig = px.bar(df_Type_suicide, x='Type', y='Total',labels={'Total':'suicides'}, height=700,width = 1400)

fig.update_layout(title_text = 'List of type and the total no of suicides')

fig.show()
df.head()
df_st_yr_tc = df.groupby(['State','Year','Type_code','Gender'])['Total'].sum()

df_st_yr_tc=pd.DataFrame(df_st_yr_tc)

df_st_yr_tc.reset_index(inplace=True)

df_st_yr_tc.head()
df_sytc_TN_AP_KL = df_st_yr_tc[df_st_yr_tc.State.isin(['Tamil Nadu','Andhra Pradesh','Kerala'])]

df_sytc_TN_AP_KL.head()
df_sytc_TN_AP_KL.State.unique()
df_st_yr_tc['Gender'].unique()
fig=px.bar(df_st_yr_tc,x='Year',y='Total',facet_col='Type_code',height=400,width=1200,color='Gender',barmode = 'group')

fig.update_layout(title_text = 'Total no of suicide in a specific type code and year')

fig.show()
df_state_year['State'].unique()
df_state_year = df_state_year[df_state_year.State != 'Total (All India)']

df_state_year = df_state_year[df_state_year.State != 'Total (States)']
df_state_year['State'].unique()
df_state_year.sort_values(by='Total',inplace=True,ascending=True)
fig=px.bar(df_state_year,x='State',y='Total',facet_col='Year',height=700,width=13000,color='Gender',barmode='group')

fig.update_layout(title_text = 'Total no of suicides in different States by year and gender')

fig.show()
df.head()


df.sort_values(by='Total',ascending=False)
df.isna().sum()
df[(df['Year']==2012)&(df['State']=='Tamil Nadu')&(df['Type']=='Love Affairs')].sum()
df.pivot_table(index='State',columns='Type',values='Total',aggfunc='sum',margins=True).head()
fig = px.bar(df_state,x='State',y='Total',labels={'Total': 'No of Suicides: '}, height=400,color='Total')

fig.update_layout(title_text = 'List of states and their total no of suicides ')

fig.show()
df_state.sort_values(by='Total',inplace=True,ascending=False)

df_state.head()
fig = px.bar(df_state[:7],x='State',y='Total',labels={'Total': 'No of Suicides: '}, height=400,color='Total')

fig.update_layout(title_text = 'List of top 7 states and their total no of suicides ')



fig.show()
df_state.sort_values(by='Total',inplace=True,ascending=True)

df_state.head()
fig =px.bar(df_state[:7],x='State',y='Total',labels={'Total': 'No of Suicides: '}, height=400,color='Total')

fig.show()
fig =px.bar(df_year_total,x='Year',y='Total',labels={'Total':'No of suicides: '}, height=400,color='Total')

fig.update_layout(title_text = 'Total no of suicides by year from 2001 - 2012')



fig.show()
fig=px.bar(df_state_year,x='State',y='Total',facet_col='Year',height=700,width=13000,color='Gender',barmode='group')

fig.update_layout(title_text = 'Suecides in states by genderand years')

fig.show()
df.head()
df_state_year_typec_gender= df.groupby(['State','Year','Type_code','Gender'])['Total'].sum()

df_state_year_typec_gender = pd.DataFrame(df_state_year_typec_gender)

df_state_year_typec_gender.reset_index(inplace = True)

df_state_year_typec_gender
df_state_year_typec_gender['State'].unique()
df_state_year_typec_gender = df_state_year_typec_gender[df_state_year_typec_gender.State != 'Total (All India)']

df_state_year_typec_gender = df_state_year_typec_gender[df_state_year_typec_gender.State != 'Total (States)']
df_state_year_typec_gender['State'].unique()
fig = px.bar(df_state_year_typec_gender,x='Year',y='Total',facet_col='Type_code',color='Gender',barmode='group',

             height=400,width=1600,hover_name='State')

fig.show()