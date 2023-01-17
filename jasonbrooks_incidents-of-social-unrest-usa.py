import pandas as pd
import numpy as np
import folium
from folium import plugins

df = pd.read_csv('../input/social-unrest/social_unrest.csv')
df = df.set_index('year')

df['Event Date'] = pd.to_datetime(df['Event Date'])
df['year'] = df['Event Date'].dt.year
df['weekday'] = df['Event Date'].dt.weekday

#CSV files for Folium map
seventeen = pd.read_csv('../input/datasets-of-social-unrest-1783-2020/1700.csv')
eighteen = pd.read_csv('../input/datasets-of-social-unrest-1783-2020/1800.csv')
nineteen = pd.read_csv('../input/datasets-of-social-unrest-1783-2020/1900.csv')
twenty = pd.read_csv('../input/datasets-of-social-unrest-1783-2020/2000.csv')
# CSV by categories
assassin = pd.read_csv('../input/datasets-of-social-unrest-1783-2020/assassin.csv')
dispute = pd.read_csv('../input/datasets-of-social-unrest-1783-2020/dispute.csv')
immigrant = pd.read_csv('../input/datasets-of-social-unrest-1783-2020/immigrant.csv')
labor = pd.read_csv('../input/datasets-of-social-unrest-1783-2020/labor.csv')
lynch = pd.read_csv('../input/datasets-of-social-unrest-1783-2020/lynch.csv')
police = pd.read_csv('../input/datasets-of-social-unrest-1783-2020/police.csv')
protest = pd.read_csv('../input/datasets-of-social-unrest-1783-2020/protest.csv')
racial = pd.read_csv('../input/datasets-of-social-unrest-1783-2020/racial.csv')
segregation = pd.read_csv('../input/datasets-of-social-unrest-1783-2020/segregation.csv')
sports = pd.read_csv('../input/datasets-of-social-unrest-1783-2020/sports.csv')
df.head(5)
#Use df.sample to get a sense of the data.
df.sample(5)
#Creating a new data type in order to plot years.
df['year'] = df['year'].astype(str)
df['decade'] = df['year'].apply(lambda x: str(x)[:3] + "0's")
df['year'] = df['year'].astype(int)
#What is the most common reason for a riot?
df['Reason'].value_counts().head(10).plot(kind = 'bar' )
#What city/ has the most incidents?
df['City'].value_counts().head(10).plot(kind = 'bar' )
# Which State?
df['State'].value_counts().head(10).plot(kind = 'bar' )
#Of the 5 most common states, what's the most common reason?
top5_states = df[df['State'].str.contains('New York|California|Illinois|Pennsylvania|Ohio')]
top5_states['Reason'].value_counts().head(9).plot(kind = 'bar')
#What were the reasons for unrest?
df['Reason'].value_counts().head(30)
#What's the most common day of the week for an event? Month? Date? Season?
df['weekday'].value_counts().sort_index().plot(kind = 'bar')
#What's the most common month for an event?
df['month'].value_counts().sort_index().plot(kind = 'bar')
#The first of the month is by far the most common day for an incident
df['day_of_month'].value_counts().head()
#In what decade has the most unrest?
df['decade'].value_counts()
#Interactive Map of Social Unrest in USA
unrest = folium.Map(location=[37.788302,-81.953481], tiles='cartodbdark_matter', zoom_start = 4)

folium.Marker(
    location=[39.8283, -98.5795],
    popup='The United States of America',
    icon=folium.Icon(color='red', icon='info-sign')
).add_to(unrest)


plugins.Fullscreen(
    position='topleft',
    title='Expand me',
    title_cancel='Exit me',
    force_separate_button=True
).add_to(unrest)


fg = folium.FeatureGroup(name='Incidents of civil unrest in the United States: 1783 - 2020')
unrest.add_child(fg)

g1 = plugins.FeatureGroupSubGroup(fg, '1783 - 1799')
unrest.add_child(g1)

g2 = plugins.FeatureGroupSubGroup(fg, '1800 - 1899')
unrest.add_child(g2)

g3 = plugins.FeatureGroupSubGroup(fg, '1900 - 1999')
unrest.add_child(g3)

g4 = plugins.FeatureGroupSubGroup(fg, '2000 - 2020')
unrest.add_child(g4)

g5 = plugins.FeatureGroupSubGroup(fg, 'Racial Attack')
unrest.add_child(g5)

g6 = plugins.FeatureGroupSubGroup(fg, 'Labor Strike')
unrest.add_child(g6)

g7 = plugins.FeatureGroupSubGroup(fg, 'Police Violence')
unrest.add_child(g7)

g8 = plugins.FeatureGroupSubGroup(fg, 'Political Protest')
unrest.add_child(g8)

g9 = plugins.FeatureGroupSubGroup(fg, 'Labor Dispute')
unrest.add_child(g9)

g10 = plugins.FeatureGroupSubGroup(fg, 'Assassination')
unrest.add_child(g10)

g11 = plugins.FeatureGroupSubGroup(fg, 'Anti-Immigrant Sentiment')
unrest.add_child(g11)

g12 = plugins.FeatureGroupSubGroup(fg, 'Lynching')
unrest.add_child(g12)

g13 = plugins.FeatureGroupSubGroup(fg, 'Sports')
unrest.add_child(g13)

g14 = plugins.FeatureGroupSubGroup(fg, 'Desegregation Protests')
unrest.add_child(g14)




#Defining the functions
def function_01(row): # 
    folium.CircleMarker(location=[row.loc['Latitude'], row.loc['Longitude']], 
                        popup=row.loc['Event Name'],tooltip= row['Reason'], 
                        radius = 5, weight = 1, color='#FFFFFF', fill_color='#e7edf7', fill_opacity= 1).add_to(g1)

seventeen.apply(function_01, axis='columns')

def function_02(row): # 
    folium.CircleMarker(location=[row.loc['Latitude'], row.loc['Longitude']], 
                        popup=row.loc['Event Name'],tooltip= row['Reason'], 
                        radius = 5, weight = 1, color='#01665e', fill_color='#01665e', fill_opacity= 1).add_to(g2)
eighteen.apply(function_02, axis='columns')

def function_03(row): # 
    folium.CircleMarker(location=[row.loc['Latitude'], row.loc['Longitude']], 
                        popup=row.loc['Event Name'],tooltip= row['Reason'], 
                        radius = 5, weight = 1, color='#35978f', fill_color='#35978f', fill_opacity= 1).add_to(g3)

nineteen.apply(function_03, axis='columns')

def function_04(row): # 
    folium.CircleMarker(location=[row.loc['Latitude'], row.loc['Longitude']], 
                        popup=row.loc['Event Name'],tooltip= row['Reason'], 
                        radius = 5, weight = 1, color='#80cdc1', fill_color='#80cdc1', fill_opacity= 1).add_to(g4)

twenty.apply(function_04, axis='columns')


#PLOTTING INCIDENTS BY REASON


# Racial Attack
def function_05(row): # 
    folium.CircleMarker(location=[row.loc['Latitude'], row.loc['Longitude']], 
                        popup=row.loc['Full Description'],tooltip= row['Event Name'], 
                        radius = 1, weight = 4, color='#8d5524', fill_color='#ffdbac', fill_opacity= 1).add_to(g5)

racial.apply(function_05, axis='columns')

#Labor Strike
def function_06(row): # 
    folium.CircleMarker(location=[row.loc['Latitude'], row.loc['Longitude']], 
                        popup=row.loc['Full Description'],tooltip= row['Event Name'], 
                        radius = 1, weight = 4, color='#a39594', fill_color='#a39594', fill_opacity= 1).add_to(g6)

labor.apply(function_06, axis='columns')

#Police Violence
def function_07(row): # 
    folium.CircleMarker(location=[row.loc['Latitude'], row.loc['Longitude']], 
                        popup=row.loc['Full Description'],tooltip= row['Event Name'], 
                        radius = 1, weight = 4, color='#e3b23c', fill_color='#e3b23c', fill_opacity= 1).add_to(g7)

police.apply(function_07, axis='columns')

#Political Protest
def function_08(row): # 
    folium.CircleMarker(location=[row.loc['Latitude'], row.loc['Longitude']], 
                        popup=row.loc['Full Description'],tooltip= row['Event Name'], 
                        radius = 1, weight = 4, color='#edebd7', fill_color='#edebd7', fill_opacity= 1).add_to(g8)

protest.apply(function_08, axis='columns')

#Labor Dispute
def function_09(row): # 
    folium.CircleMarker(location=[row.loc['Latitude'], row.loc['Longitude']], 
                        popup=row.loc['Full Description'],tooltip= row['Event Name'], 
                        radius = 1, weight = 4, color='#a09ebb', fill_color='#a09ebb', fill_opacity= 1).add_to(g9)

dispute.apply(function_09, axis='columns')

#Political Assassination
def function_10(row): # 
    folium.CircleMarker(location=[row.loc['Latitude'], row.loc['Longitude']], 
                        popup=row.loc['Full Description'],tooltip= row['Event Name'], 
                        radius = 1, weight = 4, color='#a8aec1', fill_color='#a8aec1', fill_opacity= 1).add_to(g10)

assassin.apply(function_10, axis='columns')

#Anti-Immigrant Sentiment
def function_11(row): # 
    folium.CircleMarker(location=[row.loc['Latitude'], row.loc['Longitude']], 
                        popup=row.loc['Full Description'],tooltip= row['Event Name'], 
                        radius = 1, weight = 4, color='#b5d2cb', fill_color='#b5d2cb', fill_opacity= 1).add_to(g11)

immigrant.apply(function_11, axis='columns')

#Lynching
def function_12(row): # 
    folium.CircleMarker(location=[row.loc['Latitude'], row.loc['Longitude']], 
                        popup=row.loc['Full Description'],tooltip= row['Event Name'], 
                        radius = 1, weight = 4, color='#bfffbc', fill_color='#bfffbc', fill_opacity= 1).add_to(g12)

lynch.apply(function_12, axis='columns')

#Sports
def function_13(row): # 
    folium.CircleMarker(location=[row.loc['Latitude'], row.loc['Longitude']], 
                        popup=row.loc['Full Description'],tooltip= row['Event Name'], 
                        radius = 1, weight = 4, color='#a6ffa1', fill_color='#a6ffa1', fill_opacity= 1).add_to(g13)

sports.apply(function_13, axis='columns')

#desegregation
def function_14(row): # 
    folium.CircleMarker(location=[row.loc['Latitude'], row.loc['Longitude']], 
                        popup=row.loc['Full Description'],tooltip= row['Event Name'], 
                        radius = 1, weight = 4, color='#ffffff', fill_color='#ffffff', fill_opacity= 1).add_to(g14)

segregation.apply(function_14, axis='columns')


folium.LayerControl(collapsed=False).add_to(unrest)

unrest.save("unrest.html")

unrest
