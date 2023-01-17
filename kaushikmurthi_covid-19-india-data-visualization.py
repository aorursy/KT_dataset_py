import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



import warnings

warnings.filterwarnings('ignore')



plt.style.use('fivethirtyeight')

sns.set_style('darkgrid')

plt.rcParams['font.sans-serif'] = "Arial"

plt.rcParams['font.family'] = "sans-serif"

plt.rcParams.update({'font.size': 10})
df_agegroup = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')

df_dailycases = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

df_hospbeds = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')

df_dailytests = pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingDetails.csv')

df_indiv = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')

df_popstate = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')
df_agegroup
colors = ["indianred"]

fig = px.bar(df_agegroup[df_agegroup['AgeGroup']!='Missing'],x='AgeGroup',y='TotalCases',color_discrete_sequence=colors,template='plotly_dark',

             width=800)

fig.update_layout(

    title='Total Cases per Age Group',

    font=dict(

        family="Arial",

        size=12,

        color="#7f7f7f")

    )
df_dailycases.sample(10)
# Convert date format from dd/mm/yy to mm/dd/yy

df_dailycases['Date'] = df_dailycases['Date'].apply(lambda x: '/'.join([x.split('/')[1],x.split('/')[0],x.split('/')[2]]))



# Clean state names

df_dailycases['State/UnionTerritory'] = df_dailycases['State/UnionTerritory'].apply(lambda x: 'Chhattisgarh' if x=='Chattisgarh' else x)

df_dailycases['State/UnionTerritory'] = df_dailycases['State/UnionTerritory'].apply(lambda x: 'Puducherry' if x=='Pondicherry' else x)
# Group by date

df_dailycases_date = df_dailycases.groupby('Date').sum()

df_dailycases_date.drop('Sno',axis=1,inplace=True)

df_dailycases_date.reset_index(inplace=True)

df_dailycases_date.sample(10)
# Start from March 5th 2020

start_range = int(df_dailycases_date[df_dailycases_date['Date']=='03/05/20'].index.values)

daily_growth = []

for i in range(start_range,len(df_dailycases_date)):

    diff = (df_dailycases_date['Confirmed'].loc[i]/df_dailycases_date['Confirmed'].loc[i-1])

    daily_growth.append(diff)

avg_daily_growth = round(sum(daily_growth)/len(daily_growth),2)

print('Average Daily Growth:', avg_daily_growth)
# Add 15 days to the current dataframe

predict_dates = ['04/03/20','04/04/20','04/05/20','04/06/20','04/07/20','04/08/20','04/09/20','04/10/20','04/11/20','04/12/20','04/13/20','04/14/20','04/15/20','04/16/20','04/17/20']

df_dailycases_date['Predicted_Cases'] = df_dailycases_date['Confirmed']

cols = df_dailycases_date.columns

df_dailycases_date = pd.concat([df_dailycases_date,pd.DataFrame(predict_dates,columns=['Date'])],axis=0)

df_dailycases_date = df_dailycases_date.reset_index(drop=True)

df_dailycases_date = df_dailycases_date[cols]



# Predicted cases based on average daily growth

for i in range(start_range,len(df_dailycases_date)):

    df_dailycases_date['Predicted_Cases'].loc[i] = df_dailycases_date['Predicted_Cases'].loc[i-1]*avg_daily_growth

df_dailycases_date['Predicted_Cases'] = df_dailycases_date['Predicted_Cases'].apply(lambda x: round(x))

df_dailycases_date.sample(10)
fig = px.line(df_dailycases_date.loc[start_range:],x='Date',y='Confirmed',template='plotly_dark',color_discrete_sequence=['yellow'],labels={'Date':'Date(mm/dd/yy)','Confirmed':'Number of Cases'},

             width=800,height=600)

fig.add_trace(px.scatter(df_dailycases_date.loc[start_range:],x='Date',y='Predicted_Cases',template='plotly_dark',color_discrete_sequence=['lightgreen'],labels={'Date':'Date(mm/dd/yy)'}).data[0])

fig.update_layout(

    title='Confirmed Cases in India (Actual vs Predicted)',

    font=dict(

        family="Arial",

        size=12,

        color="#7f7f7f"),

    showlegend=True

    )

fig.update_traces(marker=dict(size=8,

                              line=dict(width=2,

                                        color='DarkSlateGrey'))

                 )

fig.show()
# Group by State/Union Territory

df_dailycases_state = df_dailycases.groupby('State/UnionTerritory').max()

df_dailycases_state.drop('Sno',axis=1,inplace=True)

df_dailycases_state.reset_index(inplace=True)

df_dailycases_state.sample(10)
fig = px.scatter(df_dailycases_state[df_dailycases_state['State/UnionTerritory']!='Unassigned'],y='State/UnionTerritory',x='Confirmed',color='State/UnionTerritory',

                 color_discrete_sequence=px.colors.qualitative.Pastel,template='plotly_dark',labels={'Date':'Date(mm/dd/yy)'},

                 width=800,height=800)

fig.update_layout(

    title='Total Confirmed Cases Per State/Union Territory',

    font=dict(

        family="Arial",

        size=12,

        color="#7f7f7f"),

    showlegend=False

    )

fig.update_traces(marker=dict(size=12,

                              line=dict(width=2,

                                        color='DarkSlateGrey')),

                 marker_symbol = 'diamond')
fig = px.line(df_dailycases,x='Date',y='Confirmed',template='plotly_dark',color_discrete_sequence=px.colors.qualitative.Plotly_r,labels={'Date':'Date(mm/dd/yy)'},color='State/UnionTerritory',

             width=800,height=600)

fig.update_layout(

    title='Total Confirmed Cases Per State/Union Territory',

    font=dict(

        family="Arial",

        size=12,

        color="#7f7f7f"),

        showlegend=True

)

fig.update_traces(mode='lines+markers',

                  marker=dict(size=8,

                              line=dict(width=2,

                                        color='DarkSlateGrey'))

)
df_indiv.sample(10)
df_indiv_district = df_indiv[(df_indiv['detected_district'].notnull()) & (df_indiv['detected_state'].notnull())]

df_indiv_district.sample(5)
colors_list = ['coral','teal']

states = df_indiv_district['detected_state'].unique()

if len(states)%2==0:

    n_rows = int(len(states)/2)

else:

    n_rows = int((len(states)+1)/2)    

plt.figure(figsize=(14,60))

for idx,state in enumerate(states):    

    plt.subplot(n_rows,2,idx+1)

    y_order = df_indiv_district[df_indiv_district['detected_state']==state]['detected_district'].value_counts().index

    g = sns.countplot(data=df_indiv_district[df_indiv_district['detected_state']==state],y='detected_district',orient='v',color=colors_list[idx%2],order=y_order)

    plt.xlabel('Number of Cases')

    plt.ylabel('')

    plt.title(state)

    plt.ylim(14,-1)

plt.tight_layout()

plt.show()
df_indiv_gender = df_indiv[df_indiv['gender'].notnull()]

df_indiv_gender['gender'] = df_indiv_gender['gender'].apply(lambda gender: 'Female' if gender=='F' else 'Male')

df_indiv_gender.sample(10)
fig = px.pie(df_indiv_gender['gender'].value_counts(),values=df_indiv_gender['gender'].value_counts().values,names=df_indiv_gender['gender'].value_counts().index,

       width=800,height=500,color_discrete_sequence=px.colors.qualitative.Pastel1,labels={'value':'Total Cases'})

fig.update_layout(

    title='Total Confirmed Cases Per Gender',

    font=dict(

        family="Arial",

        size=12,

        color="#7f7f7f")

)
# Parse df_popstate and add state population to df_dailycases_state and create a new column that gives cases per capita

# Function to map population for each state

def map_population_by_state(state):

    pop = df_popstate[df_popstate['State / Union Territory']==state]['Population'].values

    if len(pop)>0:

        return float(pop/100000)

    else:

        return ''



# Add state_pop to df_dailycases_state

df_dailycases_state['State_Pop_inLakhs'] = 0

df_dailycases_state['State_Pop_inLakhs'] = df_dailycases_state['State/UnionTerritory'].apply(map_population_by_state)



# Find cases_per_capita

df_dailycases_state['Cases_Per_Capita'] = 0

df_dailycases_state['Cases_Per_Capita'] = df_dailycases_state[df_dailycases_state['State/UnionTerritory']!='Unassigned'][['Confirmed','State_Pop_inLakhs']].apply(lambda row: row.Confirmed/row.State_Pop_inLakhs,axis=1)



# View cases per capita

df_dailycases_state.drop(['Date','Time'],axis=1)
px.bar(df_dailycases_state[df_dailycases_state['State/UnionTerritory']!='Unassigned'].sort_values(by=['Cases_Per_Capita'],ascending=False),

       x='State/UnionTerritory',

       y='Cases_Per_Capita',

       template='plotly_dark',

       width=800,

       labels={'Cases_Per_Capita':'Cases Per Lakh of People'})