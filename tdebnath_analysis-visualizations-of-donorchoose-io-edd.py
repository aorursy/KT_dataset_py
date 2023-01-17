import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import plotly.plotly as py1
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

from wordcloud import WordCloud, STOPWORDS
from scipy.misc import imread
import base64
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import folium
from folium import plugins
from io import StringIO
import folium 

import missingno as msno # to view missing values
from scipy.stats import probplot
import os
print(os.listdir("../input"))
print(os.listdir("../input/usa-cities/"))
print(os.listdir("../input/io/"))
donors = pd.read_csv('../input/io/Donors.csv',low_memory=False)
donors.head()
donations = pd.read_csv('../input/io/Donations.csv')
donations.head()
resources = pd.read_csv('../input/io/Resources.csv')
resources.head()
teachers = pd.read_csv('../input/io/Teachers.csv')
teachers.head()
projects = pd.read_csv('../input/io/Projects.csv')
projects.head()
schools = pd.read_csv('../input/io/Schools.csv',error_bad_lines=False)
schools.head()
# Merge donation data with donor data 
donors_donations = donations.merge(donors, on='Donor ID', how='inner')
projects_schools = projects.merge(schools, on='School ID', how='inner')
temp = donors_donations["Donor City"].value_counts().head(30)
temp.iplot(kind='bar', xTitle = 'City name', yTitle = "Count", title = 'Top Donor cities')
donors_donations['Donation Received Date'] = pd.to_datetime(donors_donations['Donation Received Date'])
donors_donations['year'] = donors_donations['Donation Received Date'].dt.year
tempdf = donors_donations[~donors_donations.year.isin([2018])].sort_values('year') 
fig = {
    'data': [
        {
            'x': tempdf[tempdf['Donor City']==city].groupby('year').agg({'Donor ID' : 'count'}).reset_index()['year'],
            'y': tempdf[tempdf['Donor City']==city].groupby('year').agg({'Donor ID' : 'count'}).reset_index()['Donor ID'],
            'name': city, 'mode': 'expand',
        } for city in ['Chicago', 'New York', 'Brooklyn', 'Los Angeles', 'San Francisco',
       'Houston', 'Indianapolis', 'Portland', 'Philadelphia', 'Seattle',
       'Atlanta', 'Washington', 'Zionsville', 'Phoenix', 'Charlotte',
       'Columbus', 'Miami', 'Denver', 'Austin', 'Manhattan Beach', 'San Jose',
       'Dallas', 'Oklahoma City', 'Oakland', 'San Diego', 'Minneapolis',
       'Cleveland', 'Saint Louis', 'Arlington', 'Memphis',]
    ],
    'layout': {
        'title' : 'Donors Trends in Top 30 Cities (2012 - 2017)',
        'xaxis': {'title': 'Year'},
        'yaxis': {'title': "Number of Donors"},
        'showlegend':'True'
    }
}
py.iplot(fig, filename='Cities')
city_wise_donation = donors_donations.groupby('Donor City', as_index=False).agg({'Donation ID': 'count','Donation Amount':'sum'}).sort_index(by=['Donation Amount'],ascending=[False])
trace = go.Bar(
    y=city_wise_donation['Donation Amount'][:30],
    x=city_wise_donation['Donor City'][:30],
    marker=dict(
        color=city_wise_donation['Donation Amount'][:30][::-1],
        colorscale = 'reds',
        reversescale = True
    ),
)

layout = dict(
    title='Distribution Top 30 Donation Amount Cities wise',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="CityDonationAmount")
temp = donors_donations["Donor State"].value_counts().head(30)
temp.iplot(kind='bar', xTitle = 'State name', yTitle = "Count", title = 'Top Donor States', color='red')
fig = {
    'data': [
        {
            'x': tempdf[tempdf['Donor State']==state].groupby('year').agg({'Donor ID' : 'count'}).reset_index()['year'],
            'y': tempdf[tempdf['Donor State']==state].groupby('year').agg({'Donor ID' : 'count'}).reset_index()['Donor ID'],
            'name': state, 'mode': 'line',
        } for state in ['California', 'New York', 'Texas', 'Florida', 'Illinois',
       'North Carolina', 'other', 'Pennsylvania', 'Georgia', 'Massachusetts',
       'Michigan', 'Indiana', 'Virginia', 'New Jersey', 'Ohio',
       'South Carolina', 'Washington', 'Missouri', 'Arizona', 'Maryland',
       'Tennessee', 'Wisconsin', 'Connecticut', 'Colorado', 'Oregon',
       'Oklahoma', 'Minnesota', 'Alabama', 'Louisiana', 'Utah',]
    ],
    'layout': {
        'title' : 'Donors Trends in Top 30 States (2012 - 2017)',
        'xaxis': {'title': 'Year'},
        'yaxis': {'title': "Number of Donors"}
    }
}
py.iplot(fig, filename='donor_trends_states')
state_wise_donation = donors_donations.groupby('Donor State', as_index=False).agg({'Donation ID': 'count','Donation Amount':'sum'}).sort_index(by=['Donation Amount'],ascending=[False])
trace = go.Bar(
    y=state_wise_donation['Donation Amount'][:30],
    x=state_wise_donation['Donor State'][:30],
    marker=dict(
        color=state_wise_donation['Donation Amount'][:30][::-1],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Distribution Top 30 Donation Amount State wise',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="StateDonationAmount")
state_wise = donors_donations.groupby('Donor State', as_index=False).agg({'Donation ID': 'count','Donation Amount':'sum'})   
state_wise.columns = ["State","Donation_num", "Donation_sum"]
state_wise["Donation_avg"]=state_wise["Donation_sum"]/state_wise["Donation_num"]
del state_wise['Donation_num']
for col in state_wise.columns:
    state_wise[col] = state_wise[col].astype(str)
state_wise['text'] = state_wise['State'] + '<br>' +\
    'Average amount per donation: $' + state_wise['Donation_avg']+ '<br>' +\
    'Total donation amount:  $' + state_wise['Donation_sum']
state_codes = {
    'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 
    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 
    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 
    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 
    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 
    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 
    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 
    'Nevada': 'NV', 'Maine': 'ME', 'other': ''}

state_wise['code'] = state_wise['State'].map(state_codes)  
# https://plot.ly/python/choropleth-maps/
data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = state_wise['code'], # The variable identifying state
        z = state_wise['Donation_sum'].astype(float), # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = state_wise['text'], # Text to show when mouse hovers on each state
        colorbar = dict(  
            title = "Donation in USD")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = 'Donations given by different States<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
iplot(fig)
statesll=StringIO("""State,Latitude,Longitude
Alabama,32.806671,-86.791130
Alaska,61.370716,-152.404419
Arizona,33.729759,-111.431221
Arkansas,34.969704,-92.373123
California,36.116203,-119.681564
Colorado,39.059811,-105.311104
Connecticut,41.597782,-72.755371
Delaware,39.318523,-75.507141
District of Columbia,38.897438,-77.026817
Florida,27.766279,-81.686783
Georgia,33.040619,-83.643074
Hawaii,21.094318,-157.498337
Idaho,44.240459,-114.478828
Illinois,40.349457,-88.986137
Indiana,39.849426,-86.258278
Iowa,42.011539,-93.210526
Kansas,38.526600,-96.726486
Kentucky,37.668140,-84.670067
Louisiana,31.169546,-91.867805
Maine,44.693947,-69.381927
Maryland,39.063946,-76.802101
Massachusetts,42.230171,-71.530106
Michigan,43.326618,-84.536095
Minnesota,45.694454,-93.900192
Mississippi,32.741646,-89.678696
Missouri,38.456085,-92.288368
Montana,46.921925,-110.454353
Nebraska,41.125370,-98.268082
Nevada,38.313515,-117.055374
New Hampshire,43.452492,-71.563896
New Jersey,40.298904,-74.521011
New Mexico,34.840515,-106.248482
New York,42.165726,-74.948051
North Carolina,35.630066,-79.806419
North Dakota,47.528912,-99.784012
Ohio,40.388783,-82.764915
Oklahoma,35.565342,-96.928917
Oregon,44.572021,-122.070938
Pennsylvania,40.590752,-77.209755
Rhode Island,41.680893,-71.511780
South Carolina,33.856892,-80.945007
South Dakota,44.299782,-99.438828
Tennessee,35.747845,-86.692345
Texas,31.054487,-97.563461
Utah,40.150032,-111.862434
Vermont,44.045876,-72.710686
Virginia,37.769337,-78.169968
Washington,47.400902,-121.490494
West Virginia,38.491226,-80.954453
Wisconsin,44.268543,-89.616508
Wyoming,42.755966,-107.302490""")

tempdf = donors_donations.groupby(['Donor State']).agg({'Donation Amount':'sum'}).reset_index()
t1 = tempdf.sort_values('Donation Amount', ascending=False)

sdf = pd.read_csv(statesll).rename(columns={'State':'Donor State'})
sdf = sdf.merge(t1, on='Donor State', how='inner')

map4 = folium.Map(location=[39.50, -98.35], tiles='CartoDB dark_matter', zoom_start=3)
for j, rown in sdf.iterrows():
    rown = list(rown)
    folium.CircleMarker([float(rown[1]), float(rown[2])], popup=rown[0]+" $"+str(int(rown[3])), radius=float(rown[3])*0.000001, color='blue', fill=True).add_to(map4)
map4
temp = projects_schools['Project Subject Category Tree'].value_counts().head(30)
temp.iplot(kind='bar', xTitle = 'Project Subject Category', yTitle = "Count", title = 'Distribution of Project subject categories', color='green')
temp = projects_schools['Project Subject Subcategory Tree'].value_counts().head(50)
temp.iplot(kind='bar', xTitle = 'Project Subject Sub-Category', yTitle = "Count", title = 'Distribution of Project subject Sub-categories', color='blue')
temp = projects_schools['Project Resource Category'].value_counts().head(30)
temp.iplot(kind='bar', xTitle = 'Project Resource Category Name', yTitle = "Count", title = 'Distribution of Project Resource categories')
temp = schools['School Metro Type'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Distribution of school Metro Type')
cnt_srs = projects_schools['School City'].value_counts().head(20)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Distribution of School cities',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="CitySchools")
school_count = schools['School State'].value_counts().reset_index()
school_count.columns = ['state', 'schools']
for col in school_count.columns:
    school_count[col] = school_count[col].astype(str)
school_count['text'] = school_count['state'] + '<br>' + '# of schools: ' + school_count['schools']
state_codes = {
    'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 
    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 
    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 
    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 
    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 
    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 
    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 
    'Nevada': 'NV', 'Maine': 'ME', 'other': ''}

school_count['code'] = school_count['state'].map(state_codes) 
# https://plot.ly/python/choropleth-maps/
data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = school_count['code'], # The variable identifying state
        z = school_count['schools'].astype(float), # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = school_count['text'], # Text to show when mouse hovers on each state
        colorbar = dict(  
            title = "# of Schools")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = 'Number of schools in different states<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
iplot(fig)
### [Number of schools in different cities](http://)


# cities=pd.read_csv('../input/usa-cities/usa_cities.csv')
# # city_don=projects_schools.groupby('School City')['School Name'].sum().to_frame()
# city_num=schools['School City'].value_counts().to_frame()
# # city_don=city_don.merge(city_num,left_index=True,right_index=True,how='left')
# # city_don.columns=[['Amount','Donors']]
# map_cities=cities[['city','lat','lng']].merge(city_num,left_on='city',right_index=True)
# map_cities.columns=[['City','lat','lon','DonSchool Nameors']]
# map2 = folium.Map(location=[39.50, -98.35],tiles='Mapbox Control Room',zoom_start=3.5)
# locate=map_cities[['lat','lon']]
# count=map_cities['School Name']
# city=map_cities['City']
# # amt=map_cities['Amount']
# def color_producer(donors):
#     if donors < 90:
#         return 'orange'
#     else:
#         return 'green'
# for point in map_cities.index:
#     info='<b>City: </b>'+str(city.loc[point].values[0])+'<br><b>No of Donors: </b>'+str(count.loc[point].values[0])+'<br><b>Total Funds Donated: </b>'+str(amt.loc[point].values[0])+' <b>$<br>'
#     iframe = folium.IFrame(html=info, width=250, height=250)
#     folium.CircleMarker(list(locate.loc[point]),
#                         popup=folium.Popup(iframe),
# #                         radius=amt.loc[point].values[0]*0.000005,
#                         color=color_producer(count.loc[point].values[0]),
#                         fill_color=color_producer(count.loc[point].values[0]),fill=True).add_to(map2)
# map2
schools.groupby('School Metro Type')['School Percentage Free Lunch'].describe()
temp = projects['Project Grade Level Category'].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      "name": "Grade Level Category",
      #"hoverinfo":"label+percent+name",
      "hole": .5,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Distribution of Projects Grade Level Category",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Grade Level Categories",
                "x": 0.11,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
temp = donors['Donor Is Teacher'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Donor is Teacher or not')
temp = donors_donations['Donation Included Optional Donation'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Whether or not the donation included an optional donation.')

temp = projects_schools['Project Current Status'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Projects were fully funded or not.')
donors_donations = donations.merge(donors, left_on='Donor ID',right_on='Donor ID',how='left')
donors_donations_projects = donors_donations.merge(projects[['Project ID', 'School ID']],left_on='Project ID', right_on='Project ID')
donors_donations_projects__schools = donors_donations_projects.merge(schools, left_on='School ID', right_on='School ID')
donors_donations_projects__schools.head()
msno.dendrogram(donors_donations_projects__schools,orientation='top')
plt.savefig('dendrogram.png')
plt.show()
sns.distplot(donations['Donation Amount'])
# skewness
print("Skewness: %f" % donations['Donation Amount'].skew())
# kurtosis
print("Kurtosis: %f" % donations['Donation Amount'].kurt())
sns.distplot(np.log1p(donations['Donation Amount']))
plt.hist(np.log1p(donations['Donation Amount']))
# skewness
print("Skewness: %f" % np.log1p(donations['Donation Amount']).skew())
# kurtosis
print("Kurtosis: %f" % np.log1p(donations['Donation Amount']).kurt())
print('Total Donation Received',donors.shape[0])
repeating_donors=donations['Donor ID'].value_counts().to_frame()
print('Second time returning donors %: ',(repeating_donors[repeating_donors['Donor ID']>1].shape[0]/donors['Donor ID'].count())*100, '%')
print('More than 5 times returning donors %: ',(repeating_donors[repeating_donors['Donor ID']>5].shape[0]/donors['Donor ID'].count())*100, '%')
print('More than 10 times returning donors %: ',(repeating_donors[repeating_donors['Donor ID']>10].shape[0]/donors['Donor ID'].count())*100, '%')
school_states = donors_donations_projects__schools['School State'].unique()
donor_states = donors_donations_projects__schools['Donor State'].unique()

states_to_keep_mask = [x in school_states for x in donor_states]
states = donor_states[states_to_keep_mask]
donors_donations_projects__schools = donors_donations_projects__schools[
    donors_donations_projects__schools['School State'].isin(states)]
donors_donations_projects__schools = donors_donations_projects__schools[
    donors_donations_projects__schools['Donor State'].isin(states)]

donor_to_school_total_donation_statewise = donors_donations_projects__schools.pivot_table(columns='School State',
                                      index='Donor State', 
                                      values='Donation Amount', 
                                      aggfunc='sum',
                                      fill_value=0)
donor_to_school_total_donation_statewise.head()

# Take top donor states from this merged table
top_donor_states = donors_donations_projects__schools.groupby('Donor State')['Donation Amount'].sum().sort_values(ascending=False)

top_donor_states = top_donor_states[:10]

# Separate the top n donors
top_n_donors_destinations = donor_to_school_total_donation_statewise.loc[top_donor_states.index, :]

# Remove any states that none of them donate too
top_n_donors_destinations = top_n_donors_destinations.loc[:, top_n_donors_destinations.sum() > 0]

# Unpivot
donation_paths = top_n_donors_destinations.reset_index().melt(id_vars='Donor State')
donation_paths = donation_paths[donation_paths['value'] > 250000]  # Only significant amounts

# Encode state names to integers for the Sankey
donor_encoder, school_encoder = LabelEncoder(), LabelEncoder()
donation_paths['Encoded Donor State'] = donor_encoder.fit_transform(donation_paths['Donor State'])
donation_paths['Encoded School State'] = school_encoder.fit_transform(donation_paths['School State'])\
    + len(donation_paths['Encoded Donor State'].unique())
# Create a state to color dictionary
all_states = np.unique(np.array(donation_paths['School State'].unique().tolist() + donation_paths['Donor State'].unique().tolist()))
plotly_colors = ['#8424E0', '#FD28DC', '#B728FE', '#288DFF', '#00E2FB', '#ADE601', '#FCFF00', '#FEA128', '#FE0000', '#FF5203']

states_finished = False
state_colors = []
i = 0
while not states_finished:
    
    state_colors.append(plotly_colors[i]) 
    
    if len(state_colors) >= len(all_states):
        states_finished = True
        
    i += 1
    if i >= len(plotly_colors):
        i = 0
        
color_dict = dict(zip(all_states, state_colors))


sankey_labels = donor_encoder.classes_.tolist()  + school_encoder.classes_.tolist()
colors = []
for state in sankey_labels:
    colors.append(color_dict[state])

data = dict(
    type='sankey',
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(
        color = "black",
        width = 0.5
      ),
      label = sankey_labels,
      color = colors,
    ),
    link = dict(
      source = donation_paths['Encoded Donor State'],
      target = donation_paths['Encoded School State'],
      value = donation_paths['value'],
  ))

layout =  dict(
    title = "Donation Source vs Destination(Hover over to see values)",
    autosize=False,
    width=800,
    height=750,

    font = dict(
      size = 10
    )
)

fig = dict(data=[data], layout=layout)
py.iplot(fig, filename='source_destination_match.', validate=False)
donors_schools_df = donors_donations_projects__schools[donors_donations_projects__schools['Donor Is Teacher'] == 'No']
pivot2 = donors_schools_df.pivot_table(columns='School State',
                                      index='Donor State', 
                                      values='Donation Amount', 
                                      aggfunc='sum',
                                      fill_value=0)

# Scale again by the funds that state receives
sum_state_funds = donors_schools_df.groupby('School State')['Donation Amount'].sum()
pivot2 = pivot2 / sum_state_funds.transpose()
state_lat_lon = {
    'Alabama': [32.806671,-86.791130],
    'Alaska': [61.370716,-152.404419],
    'Arizona': [33.729759,-111.431221],
    'Arkansas': [34.969704,-92.373123],
    'California': [36.116203,-119.681564],
    'Colorado': [39.059811,-105.311104],
    'Connecticut': [41.597782,-72.755371],
    'Delaware': [39.318523,-75.507141],
    'District of Columbia': [38.897438,-77.026817],
    'Florida': [27.766279,-81.686783],
    'Georgia': [33.040619,-83.643074],
    'Hawaii': [21.094318,-157.498337],
    'Idaho': [44.240459,-114.478828],
    'Illinois': [40.349457,-88.986137],
    'Indiana': [39.849426,-86.258278],
    'Iowa': [42.011539,-93.210526],
    'Kansas': [38.526600,-96.726486],
    'Kentucky': [37.668140,-84.670067],
    'Louisiana': [31.169546,-91.867805],
    'Maine': [44.693947,-69.381927],
    'Maryland': [39.063946,-76.802101],
    'Massachusetts': [42.230171,-71.530106],
    'Michigan': [43.326618,-84.536095],
    'Minnesota': [45.694454,-93.900192],
    'Mississippi': [32.741646,-89.678696],
    'Missouri': [38.456085,-92.288368],
    'Montana': [46.921925,-110.454353],
    'Nebraska': [41.125370,-98.268082],
    'Nevada': [38.313515, -117.055374],
    'New Hampshire': [43.452492,-71.563896],
    'New Jersey': [40.298904,-74.521011],
    'New Mexico': [34.840515,-106.248482],
    'New York': [42.165726,-74.948051],
    'North Carolina': [35.630066,-79.806419],
    'North Dakota': [47.528912,-99.784012],
    'Ohio': [40.388783,-82.764915],
    'Oklahoma': [35.565342,-96.928917],
    'Oregon': [44.572021,-122.070938],
    'Pennsylvania': [40.590752,-77.209755],
    'Rhode Island': [41.680893,-71.511780],
    'South Carolina': [33.856892,-80.945007],
    'South Dakota': [44.299782,-99.438828],
    'Tennessee': [35.747845,-86.692345],
    'Texas': [31.054487,-97.563461],
    'Utah': [40.150032,-111.862434],
    'Vermont': [44.045876,-72.710686],
    'Virginia': [37.769337,-78.169968],
    'Washington': [47.400902,-121.490494],
    'West Virginia': [38.491226,-80.954453],
    'Wisconsin': [44.268543,-89.616508],
    'Wyoming': [42.755966,-107.302490]
}

flight_paths = []
for i in pivot2.index:
    
    for j in pivot2.columns:
        
        # Only plot if significant
        if (pivot2.loc[i, j] > 0.05) * (i != j):
               
            flight_paths.append(
                dict(
                    type = 'scattergeo',
                    locationmode = 'USA-states',                           
                    lon = [state_lat_lon[i][1], state_lat_lon[j][1]],
                    lat = [state_lat_lon[i][0], state_lat_lon[j][0]],
                    mode = 'lines',
                    line = dict(
                        width = 10 * pivot2.loc[i, j],
                        color = 'blue',                        
                    ),
                    text = '{:.2f}% of {}\'s donations come from {}'.format(100 * pivot2.loc[i, j], j, i),
                )
            )
    
layout = dict(
        title = 'Strongest out of state donation patterns (hover for details)',
        showlegend = False, 
        geo = dict(
            scope='usa',
            #projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            #subunitcolor = "rgb(217, 217, 217)",
            #countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )
    
fig = dict(data=flight_paths, layout=layout)
py.iplot(fig)


import gc
del donors_donations,projects_schools,data,temp,tempdf

gc.collect()