# import libraries
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
import plotly as py
import plotly.express as px

pd.set_option('display.max_rows', 500)
corona_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df = corona_df.copy()

df.head()
df.tail()
df.shape
df.info()
df.isnull().sum()
nan_states_df = df[df['Province/State'].isnull()]

print('nan_states_df shape is : '+ str(nan_states_df.shape))
print('nan_states_df has got : '+ str(nan_states_df['Country/Region'].nunique()) + ' unique Country/Region values')

nan_states_df = nan_states_df[['ObservationDate','Country/Region','Confirmed','Deaths','Recovered']]
nan_states_df.head()
states_df = df[df['Province/State'].notnull()]

print('states_df shape is : '+ str(states_df.shape))
print('states_df has got : '+ str(states_df['Province/State'].nunique()) + ' unique Province/State values')
 
states_df = states_df[['ObservationDate','Province/State','Country/Region','Confirmed','Deaths','Recovered']]
states_df.head()
concentrated_states_df= states_df.groupby(['ObservationDate','Country/Region'])[['Confirmed','Deaths','Recovered']].sum().reset_index()
concentrated_states_df.head()
full_countries_df = pd.concat([nan_states_df, concentrated_states_df], axis=0).reset_index()
full_countries_df.head()
lastest_full_countries_df = full_countries_df.groupby(['Country/Region'])[['ObservationDate','Confirmed','Deaths','Recovered']].max().reset_index()
lastest_full_countries_df.head()
china_df = states_df[states_df['Country/Region']=='Mainland China'] 
china_df.head()
lastest_china_df = china_df.groupby(['Province/State']).max().reset_index()
lastest_china_df.head()
other_countries_df = full_countries_df[~(full_countries_df['Country/Region']=='Mainland China')]
other_countries_df.head()
lastest_other_countries_df = other_countries_df.groupby('Country/Region')[['Confirmed','Deaths','Recovered']].max().reset_index()
lastest_other_countries_df.head()
sorted_lastest_other_countries_df = lastest_other_countries_df.sort_values(by='Confirmed', ascending=False)
lastest_other_countries_df.head()
print('Total countries affected by virus: ' + str(lastest_full_countries_df['Country/Region'].nunique()) + '\n' + 'That countries are : ' +'\n'+str(lastest_full_countries_df['Country/Region'].unique()) )
print('Worlwide Confirmed Cases: ',lastest_full_countries_df['Confirmed'].sum())
print('Worlwide Deaths: ',lastest_full_countries_df['Deaths'].sum())
print('Worlwide Recovered Cases: ',lastest_full_countries_df['Recovered'].sum())
lastest_full_countries_df.sort_values(by='Confirmed', ascending=False)
sorted_lastest_full_countries_df = lastest_full_countries_df.sort_values(by='Confirmed', ascending=False)
sorted_lastest_full_countries_df[:10]
f, ax = plt.subplots(figsize=(12, 40))

sns.set_color_codes("pastel")
sns.barplot(x="Confirmed", y="Country/Region", data=sorted_lastest_other_countries_df[:],
            label="Confirmed", color="y")

sns.set_color_codes("pastel")
sns.barplot(x="Recovered", y="Country/Region", data=sorted_lastest_other_countries_df[:],
            label="Recovered", color="g")

sns.set_color_codes("pastel")
sns.barplot(x="Deaths", y="Country/Region", data=sorted_lastest_other_countries_df[:],
            label="Deaths", color="r")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 400), ylabel="",
       xlabel="Stats")
sns.despine(left=True, bottom=True)
fig = px.pie(sorted_lastest_other_countries_df, values = 'Confirmed',names='Country/Region', height=600)
fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))

fig.show()
sorted_lastest_full_countries_df[(sorted_lastest_full_countries_df['Confirmed'] == sorted_lastest_full_countries_df['Recovered'])]
sorted_lastest_full_countries_df[(sorted_lastest_full_countries_df['Confirmed'] == sorted_lastest_full_countries_df['Deaths'])]
sorted_lastest_full_countries_df[(sorted_lastest_full_countries_df['Recovered'] < sorted_lastest_full_countries_df['Deaths'])]
lastest_full_countries_df['DeathsRatio'] = (lastest_full_countries_df['Deaths']/lastest_full_countries_df['Confirmed'])*100
lastest_full_countries_df[lastest_full_countries_df['Deaths'] > 5].sort_values(by='DeathsRatio', ascending = False)
lastest_full_countries_df['RecoveredRatio'] = (lastest_full_countries_df['Recovered']/lastest_full_countries_df['Confirmed'])*100
lastest_full_countries_df[lastest_full_countries_df['Recovered'] > 5].sort_values(by='RecoveredRatio', ascending = False)
fig = px.treemap(lastest_full_countries_df, path=["Country/Region"], values="Confirmed", 
                 title='Number of Confirmed Cases Worldwide',
                 color_discrete_sequence = px.colors.qualitative.Pastel)
fig.show()
fig = px.treemap(lastest_full_countries_df, path=["Country/Region"], values="Deaths", 
                 title='Number of Deaths Cases Worldwide',
                 color_discrete_sequence = px.colors.qualitative.Pastel)
fig.show()
fig = px.treemap(lastest_other_countries_df, path=["Country/Region"], values="Confirmed", 
                 title='Number of Confirmed Cases outside China',
                 color_discrete_sequence = px.colors.qualitative.Pastel)
fig.show()
fig = px.treemap(lastest_other_countries_df, path=["Country/Region"], values="Deaths", 
                 title='Number of Deaths Cases outside China',
                 color_discrete_sequence = px.colors.qualitative.Pastel)
fig.show()
lastest_full_countries_df['Treatment'] = (lastest_full_countries_df['Confirmed']-(lastest_full_countries_df['Recovered']+lastest_full_countries_df['Deaths']))

US = lastest_full_countries_df[lastest_full_countries_df['Country/Region']=='US'][['Treatment','Recovered','Deaths']].iloc[0]
Spain = lastest_full_countries_df[lastest_full_countries_df['Country/Region']=='Spain'][['Treatment','Recovered','Deaths']].iloc[0]
Italy = lastest_full_countries_df[lastest_full_countries_df['Country/Region']=='Italy'][['Treatment','Recovered','Deaths']].iloc[0]
France = lastest_full_countries_df[lastest_full_countries_df['Country/Region']=='France'][['Treatment','Recovered','Deaths']].iloc[0]

fig, axes = plt.subplots(
                     ncols=2,
                     nrows=2,
                     figsize=(15, 15))

ax1, ax2, ax3, ax4 = axes.flatten()

colors = ['peachpuff','salmon','tomato']
ax1.pie(US
           , colors=colors
           , autopct='%1.1f%%' # adding percentagens
           , labels=['Treatment','Recovered','Deaths']
           , shadow=True
           , startangle=140)
ax1.set_title("US Cases Distribution")

ax2.pie(Spain
           , colors=colors
           , autopct='%1.1f%%' # adding percentagens
           , labels=['Treatment','Recovered','Deaths']
           , shadow=True
           , startangle=140)
ax2.set_title("Spain Cases Distribution")

ax3.pie(Italy
        , colors=colors
        , autopct='%1.1f%%' # adding percentagens
        , labels=['Treatment','Recovered','Deaths']
        , shadow=True
        , startangle=140)
ax3.set_title("Italy Cases Distribution")

ax4.pie(France
           , colors=colors
           , autopct='%1.1f%%' # adding percentagens
           , labels=['Treatment','Recovered','Deaths']
           , shadow=True
           , startangle=140)
ax4.set_title("France Cases Distribution")

fig.legend(['Treatment','Recovered','Deaths']
           , loc = "upper right"
           , frameon = True
           , fontsize = 15
           , ncol = 2 
           , fancybox = True
           , framealpha = 0.95
           , shadow = True
           , borderpad = 1)

plt.show();

lastest_china_df.head()
print('Total states affected by virus: ' + str(lastest_china_df['Province/State'].nunique()) + '\n' + 'That countries are : ' +'\n'+str(lastest_china_df['Province/State'].unique()) )
print('China Confirmed Cases: ',lastest_china_df['Confirmed'].sum())
print('China Deaths: ',lastest_china_df['Deaths'].sum())
print('China Recovered Cases: ',lastest_china_df['Recovered'].sum())
sorted_lastest_china_df = lastest_china_df.sort_values(by='Confirmed', ascending=False)
sorted_lastest_china_df[:5]
f, ax = plt.subplots(figsize=(10,15))

sns.set_color_codes("pastel")
sns.barplot(x="Confirmed", y="Province/State", data=sorted_lastest_china_df[:],
            label="Confirmed", color="y")

sns.set_color_codes("pastel")
sns.barplot(x="Recovered", y="Province/State", data=sorted_lastest_china_df[:],
            label="Recovered", color="g")

sns.set_color_codes("pastel")
sns.barplot(x="Deaths", y="Province/State", data=sorted_lastest_china_df[:],
            label="Deaths", color="r")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 400), ylabel="",
       xlabel="Stats")
sns.despine(left=True, bottom=True)
fig = px.pie(lastest_china_df, values = 'Confirmed',names='Province/State', height=600)
fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))

fig.show()
lastest_china_df[(lastest_china_df['Confirmed'] == lastest_china_df['Recovered'])]
lastest_china_df[(lastest_china_df['Confirmed'] == lastest_china_df['Deaths'])]
lastest_china_df[(lastest_china_df['Recovered'] < lastest_china_df['Deaths'])]
lastest_china_df['DeathsRatio'] = (lastest_china_df['Deaths']/lastest_china_df['Confirmed'])*100
lastest_china_df.sort_values(by='DeathsRatio', ascending=False).head()
lastest_china_df.sort_values(by='Deaths', ascending=False).head()
lastest_china_df['RecoveredRatio'] = (lastest_china_df['Recovered']/lastest_china_df['Confirmed'])*100
lastest_china_df.sort_values(by='RecoveredRatio', ascending=False).head()
lastest_china_df.sort_values(by='Recovered', ascending=False).head()
fig = px.treemap(lastest_china_df, path=["Province/State"], values="Confirmed", 
                 title='Number of Confirmed Cases in China',
                 color_discrete_sequence = px.colors.qualitative.Pastel)
fig.show()
fig = px.treemap(lastest_china_df, path=["Province/State"], values="Deaths", 
                 title='Number of Deaths Cases in China',
                 color_discrete_sequence = px.colors.qualitative.Pastel)
fig.show()
temp = sorted_lastest_china_df[sorted_lastest_china_df['Confirmed'] > 1000]
temp['Treatment'] = (temp['Confirmed']-(temp['Recovered']+temp['Deaths']))
temp.head()
Hubei = temp[temp['Province/State']=='Hubei'][['Recovered','Deaths','Treatment']].iloc[0]
Guangdong = temp[temp['Province/State']=='Guangdong'][['Recovered','Deaths','Treatment']].iloc[0]
Henan = temp[temp['Province/State']=='Henan'][['Recovered','Deaths','Treatment']].iloc[0]
Zhejiang = temp[temp['Province/State']=='Zhejiang'][['Recovered','Deaths','Treatment']].iloc[0]
Hunan = temp[temp['Province/State']=='Hunan'][['Recovered','Deaths','Treatment']].iloc[0]

fig, axes = plt.subplots(
                     ncols=3,
                     nrows=2,
                     figsize=(13,15))

ax1,ax2,ax3,ax4,ax5,ax6 = axes.flatten()


labels = ['Treatment','Recovered','Deaths']
explode = (0, 0,0)

ax1.pie(Hubei,explode=explode,labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=180)
ax1.set_title('Hubei')
  
ax2.pie(Guangdong,explode=explode,labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=180)
ax2.set_title('Guangdong')

ax3.pie(Henan,explode=explode,labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=180)
ax3.set_title('Henan')

ax4.pie(Zhejiang,explode=explode,labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=180)
ax4.set_title('Zhejiang')

ax5.pie(Hunan,explode=explode,labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=180)
ax5.set_title('Hunan')

plt.show()
full_countries_df.head()
fig = px.choropleth(full_countries_df, 
                    locations="Country/Region", 
                    locationmode = "country names",
                    color="Confirmed",
                    #color_continuous_scale='Rainbow',
                    hover_name="Country/Region", 
                    animation_frame="ObservationDate"
                   )

fig.update_layout(
    title_text = 'Spread of Coronavirus',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()
temp = full_countries_df.groupby('ObservationDate')['Country/Region'].nunique().reset_index()
temp.columns = ['ObservationDate','CountOfCountry']
fig = px.bar(temp, x='ObservationDate', y='CountOfCountry')
fig.update_layout(
    title_text = 'Number Of Countries With Cases',
    title_x = 0.5)
fig.show()
line_data = full_countries_df.groupby('ObservationDate').sum().reset_index()

line_data = line_data.melt(id_vars='ObservationDate', 
                 value_vars=['Confirmed',
                             'Deaths',
                             'Recovered', 
                             ], 
                 var_name='Ratio', 
                 value_name='Value')

fig = px.line(line_data, x="ObservationDate", y="Value", color='Ratio', 
              title='Confirmed cases, Recovered cases, and Death Over Time')
fig.show()
temp = china_df.groupby('ObservationDate')['Province/State'].nunique().reset_index()
temp.columns = ['ObservationDate','CountOfState']
fig = px.bar(temp, x='ObservationDate', y='CountOfState')
fig.update_layout(
    title_text = 'Number Of Countries With Cases',
    title_x = 0.5)
fig.show()
temp = china_df.groupby('ObservationDate').sum().reset_index()

temp = temp.melt(id_vars='ObservationDate', 
                 value_vars=['Confirmed', 
                             'Deaths', 
                             'Recovered'], 
                 var_name='Ratio', 
                 value_name='Value')

fig = px.line(temp, x="ObservationDate", y="Value", color='Ratio', 
              title='Confirmed cases, Recovered cases, and Death Over Time In China')
fig.show()