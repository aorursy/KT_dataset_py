#importing the required libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
#reading the csv
conference = pd.read_csv("../input/2020-conferences-cancelled-due-to-coronavirus/2020 Conferences Cancelled Due to Coronavirus - Sheet1.csv")
#looking at the dataset
conference.head()
#looking at the dataset general info
conference.info()
#checking for null values
conference.isnull().sum()
#fetching only the month name from the dataset
conference[['Date1','Date2','Date3']] = conference['Scheduled date of physical event'].str.split(expand=True)
conference.drop(['Scheduled date of physical event','Date2','Date3'], axis=1, inplace=True)
#fetching only the month name from the dataset
conference[['Month','Date']] = conference['Date1'].str.split("-", expand=True)
conference.drop(['Date1','Date'], axis=1, inplace=True)
#replacing 'Mar' by 'March' to have consistency in our data
conference.Month[conference.Month == 'Mar'] = 'March'
conference
#looking at different values in Status column
print(conference['Status'].nunique())
print(conference['Status'].unique())
#creating a dataframe using Status column and customizing its background to highlight the numbers
status_count = conference['Status'].value_counts()
#status_count

df_status = pd.DataFrame(status_count)
df_status = df_status.reset_index()
df_status.columns = ['Status', 'Count']

df_status.style.background_gradient(cmap = 'Blues')
#interactive graph to visualize the status of conferences scheduled worldwide
labels = df_status.iloc[:,0]
values = df_status.iloc[:,1]

fig1 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
fig1.update_layout(title ="Status of conferences worldwide")
fig1.show()
#Complete info about the cancelled conferences
cancelled = conference[conference['Status'] == 'Cancelled']
cancelled.style.set_properties(**{'background-color':'#eca1a6'})
#Complete info about the postponed conferences
postponed = conference[conference['Status'] == 'Postponed']
postponed.style.set_properties(**{'background-color':'#878f99'})
#Complete info about the online conferences
online = conference[conference['Status'] == 'Online']
online.style.set_properties(**{'background-color':'#80ced6'})
#creating a dataframe using Month column and customizing its background to highlight the numbers
conference_month = conference['Month'].value_counts()
conference_month = pd.DataFrame(conference_month)
conference_month = conference_month.reset_index()
conference_month.columns = ['Month','Count']
conference_month.style.background_gradient(cmap='YlGn')
#interactive graph to visualize the status of conferences scheduled venue-wise in the United States
fig2 = go.Figure(data=go.Scatter(x=conference_month['Month'], y=conference_month['Count'], 
                                mode='markers',
                                marker=dict(size=16, color=np.random.randn(10), colorscale='Viridis')))
fig2.update_layout(title ="Month-wise scheduled conferences in the world")
fig2.show()
#looking at different values in Country column
print(conference['Country'].nunique())
print(conference['Country'].unique())
#creating a dataframe using Country column and customizing its background to highlight the numbers
country_count = conference['Country'].value_counts()
#country_count

df_country = pd.DataFrame(country_count)
df_country = df_country.reset_index()
df_country.columns = ['Country','Count']

df_country.style.background_gradient(cmap = 'Reds')
#interactive graph to visualize the conferences scheduled country-wise
fig3 = px.bar(df_country, x ='Country', y='Count')
fig3.update_layout(title ="Country-wise scheduled conferences")
fig3.show()
#info of the conference status country-wise
crosstab1 = pd.crosstab(conference['Country'], conference['Status'])
crosstab1 = crosstab1.reset_index()
crosstab1.columns = ['Country','Cancelled','Online','Postponed']
crosstab1.style.background_gradient('Reds')
#interactive graph to visualize the status of conferences scheduled country-wise in the United States
fig4 = go.Figure(data=[go.Bar(name='Cancelled', x=crosstab1['Country'], y=crosstab1['Cancelled']),
                      go.Bar(name='Online', x=crosstab1['Country'], y=crosstab1['Online']),
                      go.Bar(name='Postponed', x=crosstab1['Country'], y=crosstab1['Postponed'])])
fig4.update_layout(barmode='group', title="Country-wise status of conferences")
fig4.show()
#looking at different values in Venue column
print(conference['Venue'].nunique())
print(conference['Venue'].unique())
#creating a dataframe using Venue column and customizing its background to highlight the numbers
venue_count = conference['Venue'].value_counts()
#status_count

df_venue = pd.DataFrame(venue_count)
#df_status

df_venue.style.background_gradient(cmap = 'Purples')
#info of the conference status Venue-wise
crosstab2 = pd.crosstab(conference['Venue'], conference['Status'])
crosstab2.style.background_gradient('Oranges') 
#US had the maximum conferences so looking at the US data closely
US_conference = conference[conference['Country'] == 'US']
US_venue_conference = pd.DataFrame(US_conference['Venue'].value_counts())
US_venue_conference = US_venue_conference.reset_index()
US_venue_conference.columns = ['Venue', 'Count']
US_venue_conference.style.background_gradient('Greys') 
#interactive graph to visualize the conferences scheduled venue-wise in the United States
fig5 = px.bar(US_venue_conference, x ='Venue', y='Count', color='Count')
fig5.update_layout(title ="Venue-wise scheduled conferences in the United States")
fig5.show()
#info of the conference status Venue-wise
crosstab3 = pd.crosstab(US_conference['Venue'], US_conference['Status'])
crosstab3 = crosstab3.reset_index()
crosstab3.columns = ['Venue','Cancelled','Online','Postponed']
crosstab3.style.background_gradient('GnBu')
#interactive graph to visualize the status of conferences scheduled venue-wise in the United States
fig6 = go.Figure(data=[go.Bar(name='Cancelled', x=crosstab3['Venue'], y=crosstab3['Cancelled']),
                      go.Bar(name='Online', x=crosstab3['Venue'], y=crosstab3['Online']),
                      go.Bar(name='Postponed', x=crosstab3['Venue'], y=crosstab3['Postponed'])])
fig6.update_layout(barmode='group', title ="Venue-wise status of conferences in the United States")
fig6.show()