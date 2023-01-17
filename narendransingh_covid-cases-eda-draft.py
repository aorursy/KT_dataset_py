import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import math 
%matplotlib inline
df = pd.read_csv('../input/covid19-country-level-data-for-epidemic-model/Country_Level_Info.csv')
df.head()
df.tail()
df.shape
df.isna().sum()
df.info()
df.describe(include='int')
df.describe(include='object')
filtered_df = df[df['alpha-3_code'].isnull()]
filtered_df
df_code.shape
filtered_df['Country_Region'].value_counts()
df=df.drop(['alpha-3_code'],axis=1)
df.columns
df.corr()
df.groupby(['Country_Region']).mean().round(decimals=4)
df['Date'] = pd.to_datetime(df['Date'],format='%Y/%m/%d')
df['Date'].dtype
df['Date_Month'] = df['Date'].map(lambda x: x.strftime('%m'))

df['Date_Year'] = df['Date'].map(lambda x: x.strftime('%Y'))

df['Date_Date'] = df['Date'].map(lambda x: x.strftime('%m%d'))

#df
df['Date_Month'].dtype
df_filtered_country=df[df['Country_Region']=='India']

#df_filtered_country#.sort_values('Total_Confirmed_Cases',ascending=False)

#df_filtered_by_month=df[df['Date_Month']== '02']

#df_filtered_month_country=df_filtered__by_month[df_filtered__by_month['Country_Region']== 'US']

#df_filtered_month_country
group_date_month = df_filtered_country.groupby(['Date_Month','Country_Region']).max()#.reset_index()

group_date_month

#group_date_month.get_group('Afghanistan')

#group_date_month.(lambda x: x['Country_Region'] == 'Afghanistan')
# for whole dataset

group_date_month = df.groupby(['Date_Month','Country_Region']).max()#.reset_index()

#group_date_month.get_group('Afghanistan')

#group_date_month.apply(lambda x: x['Country_Region'] == 'Afghanistan', axis=1)

group_date_month
#Create sorted data table for Countries by Total confirmed Cases

sorted_data=df.groupby('Country_Region').max().reset_index()

sorted_data=sorted_data.sort_values('Total_Confirmed_Cases', ascending=False)

sorted_data = sorted_data.head(20)

sorted_data
# Create a plot

plt.figure(figsize=(25,8))

# Add title
plt.title("Total cases by Country")

#grouped_data=df.groupby('Country_Region').sum().reset_index()

#grouped_data=grouped_data.sort_values('Total_Confirmed_Cases', ascending=False)
#grouped_data
#y_pos = np.arange(len(bars))

g = sns.barplot(x="Total_Confirmed_Cases", y="Country_Region", data=sorted_data,
            color="b",label="Total")
           
    
#label="Total", aspect=.7,hue = "Country_Region",orient="v" , #errcolor='.26'
#sns.barplot(x = grouped_data['Total_Confirmed_Cases'], y=grouped_data)
#df['Total_Confirmed_Cases'].sum()
sorted_data1=sorted_data.sort_values('Total_Confirmed_Cases', ascending=False)

fig = px.sunburst(
    data_frame = sorted_data1,
    path = ['Country_Region', 'Total_Confirmed_Cases', 'Total_Recovered_Cases'],
    values = 'Total_Confirmed_Cases',
    hover_name= 'Country_Region',
    #color = "Country_Region",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    maxdepth = -1,
    branchvalues='remainder'
)

fig.update_traces(textinfo='label+percent root')
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))


fig.show()
#ax = sns.scatterplot(x=df['date_month'], y=df['Total_Confirmed_Cases'], data=df)

temp =df[df['Country_Region']=='Italy']

plt.title("Active Cases in Italy")

ax = sns.lineplot(x=temp.Date_Month, y=temp.Remaining_Confirmed_Cases, data=temp)

ax.set(xlabel='Months', ylabel='Active Cases')

plt.show()
total = df.groupby('Date')['Total_Confirmed_Cases', 'Total_Fatalities', 
                           'Total_Recovered_Cases', 'Remaining_Confirmed_Cases'].sum().reset_index()

total = total[total['Date'] == max(total['Date'])].reset_index(drop=True)

total

#Acive cases(remaining_Confirm_cases) = Total cases - Total Death - Total Recovery
plt.figure(figsize= (40,10))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 10)

#plt.xlabel('Day by Day',fontsize = 30)

#plt.ylabel('Total_Confirmed_Cases',fontsize = 30)

plt.title("Worldwide Confirmed Cases Over Time" , fontsize = 30)

total_cases = df.groupby('Date')['Date', 'Total_Confirmed_Cases'].max()#.reset_index()

#total_cases['Date'] = pd.to_datetime(total_cases['Date'])


ax = sns.pointplot( x = total_cases.Date.dt.date ,y = total_cases.Total_Confirmed_Cases )#, color = 'r')

ax.set(xlabel='Day by Day', ylabel='Total_Confirmed_Cases')
plt.show()

sorted_data_active = sorted_data.sort_values('Remaining_Confirmed_Cases', ascending=False)

sorted_data_active = sorted_data_active.head(35)

#sorted_data_active
fig = px.bar(sorted_data_active, x='Country_Region', y='Remaining_Confirmed_Cases', 
             text='Remaining_Confirmed_Cases', title = 'Most Active Cases around the World')

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')#, yaxis=dict(title='USD (millions)')

fig.show()
#py.offline.iplot(fig)
sorted_data_deaths=sorted_data.sort_values('Total_Fatalities', ascending=False)

sorted_data_death = sorted_data_deaths.head(35)

#sorted_data_death
fig = px.bar(sorted_data_death, x='Country_Region', y='Total_Fatalities', 
             text='Total_Fatalities', title = 'Most Death Cases around the World')

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
total_new_cases = df.groupby('Country_Region')['Country_Region', 
                                               'Total_Recovered_Cases', 'Total_Confirmed_Cases','Total_Fatalities'].max()#.reset_index()
total_new_cases=total_new_cases.sort_values('Total_Confirmed_Cases', ascending = False)
total_new_cases=total_new_cases.head(35)
#total_new_cases
fig = go.Figure()

fig.add_trace(go.Bar(x=total_new_cases.Country_Region,
                y=total_new_cases.Total_Confirmed_Cases,
                name='Confirmed',
                marker_color='rgb(55, 83, 109)',
                #hovertemplate = 'Confirmed:%{y:.2f}',
                ))
fig.add_trace(go.Bar(x=total_new_cases.Country_Region,
                y=total_new_cases.Total_Recovered_Cases,
                name='Recovered',
                marker_color='rgb(26, 118, 255)'
                ))

fig.add_trace(go.Bar(x=total_new_cases.Country_Region,
                y=total_new_cases.Total_Fatalities,
                name='Deaths',
                marker_color='rgb(200, 20, 25)'
                     
                ))

fig.update_layout(
    title='Total Cases vs. Total Recovered',
    xaxis_tickfont_size=10,
    yaxis=dict(
        title='COVID-19 Cases',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=1,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)

fig.show()
temp = df.groupby('Country_Region')['Country_Region', 'Total_Fatalities', 'Total_Recovered_Cases', 
                                    'Total_Confirmed_Cases'].max()

temp['Rate_of_Deaths'] = round(temp['Total_Fatalities']/temp['Total_Confirmed_Cases'], 3)*100
temp['Rate_of_Recovery'] = round(temp['Total_Recovered_Cases']/temp['Total_Confirmed_Cases'], 3)*100

temp = temp.sort_values('Total_Confirmed_Cases', ascending=False)

temp = temp.head(35)

#temp
fig = go.Figure()

fig.add_trace(go.Scatter(x=temp.Country_Region,
                         y=temp.Rate_of_Recovery,
                         mode='lines',
                         name='Recovery Rate',
                         marker_color='rgb(0, 20, 20)'))

fig.add_trace(go.Scatter(x=total_new_cases.Country_Region,
                     y=temp.Rate_of_Deaths,
                     mode='lines',
                     name='Death Rate',
                     marker_color='rgb(255, 0, 0)' ))

#data = [trace1, trace2]
fig.update_layout(
    title='Deaths vs Recovery based for the Countries having highest number of Cases',
    xaxis_tickfont_size=10,
    yaxis=dict(
        title='Percentage',
        zeroline = True,
        showline = True,
        titlefont_size=16,
        tickfont_size=14))
              
#fig = dict(data = data, layout = layo`aaut)

#fig.show()




