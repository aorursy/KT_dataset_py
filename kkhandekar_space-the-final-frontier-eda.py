# Generic

import numpy as np

import pandas as pd

import gc



# Plot

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import plotly

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

plotly.offline.init_notebook_mode (connected = True)

# Load Data

url = '../input/all-space-missions-from-1957/Space_Corrected.csv'

df=pd.read_csv(url,index_col=0)

df.drop(['Unnamed: 0.1','Detail'],axis=1,inplace=True)



# Rename Columns

df.rename(columns={'Company Name':"Company", 'Status Rocket':"Rocket_Status",

                   ' Rocket':"RocketCost", 'Status Mission':"Mission_Status", "Datum":"Date"}, inplace=True)

# Extracting Date

df['Date']=pd.to_datetime(df['Date'])

df['Date']=df['Date'].astype(str)

df['Date']=df['Date'].str.split(' ',expand=True)[0]

df['Date']=pd.to_datetime(df['Date'])



# Rocket Cost

df['RocketCost']=df['RocketCost'].str.replace(',','')

df['RocketCost']=df['RocketCost'].astype(float)

df.dropna(inplace=True)  # dropping null values



# Rocket Status

df['Rocket_Status']=df['Rocket_Status'].str.replace('Status','')



# Extracting Country from Location

df['Country']=df['Location'].str.split(', ').str[-1]

df.drop('Location', axis=1, inplace=True)



df.head()
# Replace

loc = ['Kazakhstan','Shahrud Missile Test Site','New Mexico', 'Yellow Sea', 'Pacific Missile Range Facility',

       'Pacific Ocean','Barents Sea','Gran Canaria','Kenya']

cntry = ['Russia','Iran','USA','China','USA','Russia','USA','USA','Italy']



for x,y in zip(loc,cntry):

    df.replace(to_replace=x, value=y, inplace=True)
df_status=df.Mission_Status.value_counts().reset_index()



fig=px.pie(df_status,values='Mission_Status',names='index', color_discrete_sequence=px.colors.sequential.RdPu, hover_data=['index'],

           labels={'Mission_Status':'Count', 'index':'Status'})



fig.update_layout(title='Mission status',title_x=0.5)

fig.update_traces(textfont_size=15,textinfo='percent')

fig.show()
df_status=df.Rocket_Status.value_counts().reset_index()



fig=px.pie(df_status,values='Rocket_Status',names='index', color_discrete_sequence=px.colors.sequential.RdPu, hover_data=['index'],

           labels={'Rocket_Status':'Count', 'index':'Status'})



fig.update_layout(title='Rocket Status',title_x=0.5)

fig.update_traces(textfont_size=15,textinfo='percent')

fig.show()

df['Count']=1

df_comps=df.groupby(['Company','Mission_Status'])['Count'].sum().reset_index().sort_values(by='Count',ascending=False)

df_comps = df_comps.head(10)

fig = px.bar(df_comps, x="Company", y="Count", color="Mission_Status", title="Top 10 Companies in Space(with Mission Status)", 

             text='Count',color_discrete_sequence=px.colors.sequential.turbid)



fig.show()
df['Count']=1

df_comps=df.groupby(['Company','Rocket_Status'])['Count'].sum().reset_index().sort_values(by='Count',ascending=False)

df_comps = df_comps.head(10)

fig = px.bar(df_comps, x="Company", y="Count", color="Rocket_Status", title="Top 10 Companies in Space(with Rocket Status)", 

             text='Count',color_discrete_sequence=px.colors.sequential.turbid)



fig.show()
df['Count']=1

df_comps=df.groupby(['Company','RocketCost'])['Count'].sum().reset_index().sort_values(by='Count',ascending=False)

df_comps = df_comps.head(10)

fig = px.bar(df_comps, x="Company", y="Count", color="RocketCost", title="Top 10 Companies in Space ( with Combined Mission Cost(million$) )", 

             text='Count',color_discrete_sequence=px.colors.sequential.Peach)



fig.show() 
df['Count']=1

df_cntry=df.groupby(['Country','Mission_Status'])['Count'].sum().reset_index().sort_values(by='Count',ascending=False)

df_cntry = df_cntry.head(20)

fig = px.bar(df_cntry, x="Country", y="Count", color="Mission_Status", title="Top 10 Countries in Space(with Mission Status)", 

             text='Count',color_discrete_sequence=px.colors.sequential.turbid)



fig.show()
df['Count']=1

df_cntry=df.groupby(['Country','RocketCost'])['Count'].sum().reset_index().sort_values(by='Count',ascending=False)

df_cntry = df_cntry.head(20)

fig = px.bar(df_cntry, x="Country", y="Count", color="RocketCost", title="Top Countries in Space ( with Combined Mission Cost(million$) )", 

             text='Count',color_discrete_sequence=px.colors.sequential.Peach)



fig.show() 
df['Year']=df['Date'].dt.year

df_year=df.groupby('Year')['Count'].sum().reset_index()
fig=px.line(df_year,y='Count',x='Year',height=800,width=1000)

fig.update_layout(title='Number of missions per year',font_size=20,title_x=0.5)

fig.show()

df_year=df.groupby(['Year','Country'])['Count'].sum().reset_index()



fig = px.scatter(df_year, x="Year", y="Count", color="Country", size='Count',

                 hover_data=['Country','Year'], log_x=True, size_max=60, title="Number of Missions per Country")

fig.show()
df_year=df.groupby(['Year','Mission_Status'])['Count'].sum().reset_index()



fig = px.scatter(df_year, x="Year", y="Count", color="Mission_Status", size='Count',

                 hover_data=['Mission_Status','Year'], log_x=True, size_max=60, title="Mission Status per Country")

fig.show()
# Creating a seperate DF for selected four state owned Space Companies

df_nasa=df[df['Company']=='NASA']

df_isro=df[df['Company']=='ISRO']

df_casc=df[df['Company']=='CASC']

df_rosc=df[df['Company']=='Roscosmos']
fig1=plt.figure(figsize=(20,15))



fig1.suptitle("The Selected State Owned Companies & their Success Rate", fontsize=20)



ax1=fig1.add_subplot(221)

sns.countplot('Mission_Status',data=df_nasa,ax=ax1, palette='BuGn_r')

ax1.set_title('NASA success rate = {0:.2f}%'.format(100*df_nasa['Mission_Status'].value_counts()[0]/df_nasa.shape[0]),size=12)



ax2=fig1.add_subplot(222)

sns.countplot('Mission_Status',data=df_rosc,ax=ax2,palette='Blues')

ax2.set_title('Roscosmos success rate = {0:.2f}%'.format(100*df_rosc['Mission_Status'].value_counts()[0]/df_rosc.shape[0]),size=12)





ax3=fig1.add_subplot(223)

sns.countplot('Mission_Status',data=df_casc,ax=ax3,palette=sns.color_palette("cubehelix", 8))

ax3.set_title('CASC success rate = {0:.2f}%'.format(100*df_casc['Mission_Status'].value_counts()[0]/df_casc.shape[0]),size=12)





ax4=fig1.add_subplot(224)

sns.countplot('Mission_Status',data=df_isro,ax=ax4,palette=sns.cubehelix_palette(8))

ax4.set_title('ISRO success rate = {0:.2f}%'.format(100*df_rosc['Mission_Status'].value_counts()[0]/df_isro.shape[0]),size=12)

fig1=plt.figure(figsize=(20,15))



fig1.suptitle("The Selected State Owned Companies & their Rocket Status", fontsize=20)



ax1=fig1.add_subplot(221)

sns.countplot('Rocket_Status',data=df_nasa,ax=ax1, palette='BuGn_r')

ax1.set_title('NASA Rocket Status',size=12)



ax2=fig1.add_subplot(222)

sns.countplot('Rocket_Status',data=df_rosc,ax=ax2,palette='Blues')

ax2.set_title('Roscosmos Rocket Status',size=12)





ax3=fig1.add_subplot(223)

sns.countplot('Rocket_Status',data=df_casc,ax=ax3,palette=sns.color_palette("cubehelix", 8))

ax3.set_title('CASC Rocket Status',size=12)





ax4=fig1.add_subplot(224)

sns.countplot('Rocket_Status',data=df_isro,ax=ax4,palette=sns.cubehelix_palette(8))

ax4.set_title('ISRO Rocket Status',size=12)



fig1=plt.figure(figsize=(20,15))



fig1.suptitle("The Selected State Owned Companies & their Launch Year", fontsize=20)



ax1=fig1.add_subplot(221)

sns.lineplot(x="Year", y="Count", data=df_nasa.groupby('Year')['Count'].sum().reset_index(), ax=ax1,color='maroon')

ax1.set_title('NASA Launch Years',size=12)



ax2=fig1.add_subplot(222)

sns.lineplot(x="Year", y="Count", data=df_rosc.groupby('Year')['Count'].sum().reset_index(), ax=ax2,color='darkolivegreen')

ax2.set_title('Roscosmos Launch Years',size=12)



ax3=fig1.add_subplot(223)

sns.lineplot(x="Year", y="Count", data=df_casc.groupby('Year')['Count'].sum().reset_index(), ax=ax3,color='darkslategrey')

ax3.set_title('CASC Launch Years',size=12)



ax4=fig1.add_subplot(224)

sns.lineplot(x="Year", y="Count", data=df_isro.groupby('Year')['Count'].sum().reset_index(), ax=ax4,color='navy')

ax4.set_title('ISRO Launch Years',size=12)

fig1=plt.figure(figsize=(20,15))



fig1.suptitle("The Selected State Owned Companies & their Mission Costs over the Years", fontsize=20)



ax1=fig1.add_subplot(221)

sns.lineplot(x="Year", y="RocketCost", data=df_nasa.groupby('Year')['RocketCost'].sum().reset_index(), ax=ax1,color='maroon')

ax1.set_title('NASA Mission Costs',size=12)



ax2=fig1.add_subplot(222)

sns.lineplot(x="Year", y="RocketCost", data=df_rosc.groupby('Year')['RocketCost'].sum().reset_index(), ax=ax2,color='darkolivegreen')

ax2.set_title('Roscosmos Mission Costs',size=12)



ax3=fig1.add_subplot(223)

sns.lineplot(x="Year", y="RocketCost", data=df_casc.groupby('Year')['RocketCost'].sum().reset_index(), ax=ax3,color='darkslategrey')

ax3.set_title('CASC Mission Costs',size=12)



ax4=fig1.add_subplot(224)

sns.lineplot(x="Year", y="RocketCost", data=df_isro.groupby('Year')['RocketCost'].sum().reset_index(), ax=ax4,color='navy')

ax4.set_title('ISRO Mission Costs',size=12)