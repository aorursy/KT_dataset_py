# To store the data
import pandas as pd

# To do linear algebra
import numpy as np

# To create plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# To create nicer plots
import seaborn as sns

# To create interactive maps
from plotly.offline import init_notebook_mode, iplot
import plotly.offline as offline
from plotly import tools
init_notebook_mode(connected=True)
offline.init_notebook_mode()

# To create interactive plots
from bokeh.models import ColumnDataSource, HoverTool, WheelZoomTool, ResetTool, PanTool
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()

# To sort dictionaries
import operator

# To generate word clouds
from wordcloud import WordCloud

# Load the data
df = pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')
# Parse the date and set the index
df.date = pd.to_datetime(df.date)
df.set_index('date', inplace=True)

# Display
print('The Dataset has {} entries with {} features.'.format(df.shape[0], df.shape[1]))
print('Here is an example entry:')
df.head(1)
# Group by day, count incidens, plot graph
incident_df = df.groupby(pd.Grouper(freq='d')).agg({'incident_id':'count'}).rename(columns={'incident_id':'incidents'})
incident_df.plot(figsize=(16,5), title='Daily incidents in the US', color='#ff4500')

text = 'The data seems\nunreliable\nbefore 2014'
plt.annotate(text, (incident_df.index[364], incident_df.incidents[364]), xytext=(-120, 50), textcoords='offset points', arrowprops=dict(facecolor='black', shrink=0.05))
plt.xlabel('Date')
plt.ylabel('Incidents')
plt.show()
# Exclude the unreliable data
df = df.loc['2014':]
print('The remaining DataFrame as {} entries.'.format(df.shape[0]))
# Check for missing values, compute the percentage, reverse the series, plot the graph
df.isna().mean().mul(100).iloc[::-1].plot(kind='barh', figsize=(14,5), grid=True, title='Percentage of missing values for each feature', color='#ff4500')
plt.xlabel('Percentage')
plt.ylabel('Feature')
plt.show()
f, axarr = plt.subplots(4, sharex=True, figsize=(14,6))
df.groupby(pd.Grouper(freq='d')).n_killed.sum().plot(ax=axarr[0], color='#ff4500')
df.groupby(pd.Grouper(freq='w')).n_killed.sum().plot(ax=axarr[1], color='#ff4500')
df.groupby(pd.Grouper(freq='m')).n_killed.sum().plot(ax=axarr[2], color='#ff4500')
df.loc['2014':'2017'].groupby(pd.Grouper(freq='y')).n_killed.sum().plot(ax=axarr[3], color='#ff4500')
axarr[0].set_title('Incidents grouped by day')
axarr[1].set_title('Incidents grouped by week')
axarr[2].set_title('Incidents grouped by month')
axarr[3].set_title('Incidents grouped by year')
axarr[0].set_ylim([0, 100])
axarr[1].set_ylim([0, 400])
axarr[2].set_ylim([0, 1500])
axarr[3].set_ylim([0, 17000])
plt.tight_layout()
plt.show()
# Group by year, sum n_killed, compute percentage difference, plot graph
year_df = df.loc['2014':'2017'].groupby(pd.Grouper(freq='y')).n_killed.sum()
((year_df / year_df.values[0] - 1) * 100).plot(kind='bar', grid=True, figsize=(14,5), title='Percentage difference of deaths to 2014', color='#ff4500')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(range(0, len(year_df)), year_df.index.year, rotation=0)
plt.show()
# Compute the DataFrame grouped by weekday
weekday_df = df.groupby(df.index.weekday).agg({'incident_id':'count', 'n_killed':['mean', sum]})

# Get the weekdays
weekday_dict = {day:name for day, name in zip(df.index.weekday.unique(), df.index.weekday_name.unique())}
_, days = zip(*sorted(weekday_dict.items(), key=operator.itemgetter(0)))

# Rename hierachical columns
weekday_df.columns = ['_'.join([first, second]) for first, second in zip(weekday_df.columns.get_level_values(0), weekday_df.columns.get_level_values(1))]

# Create subplots
f, axarr = plt.subplots(3, sharex=True, figsize=(14,6))
weekday_df.incident_id_count.plot(ax=axarr[0], grid=True, color='#baff00', linewidth=5)
weekday_df['n_killed_sum'].rename(columns={'n_killed_sum':'killed'}).plot(ax=axarr[1], grid=True, color='#00baff', linewidth=5)
weekday_df['n_killed_mean'].rename(columns={'n_killed_mean':'killed'}).plot(ax=axarr[2], grid=True, color='#4400ff', linewidth=5)
axarr[0].set_title('Incidents count')
axarr[1].set_title('Killed sum')
axarr[2].set_title('Killed mean')
axarr[0].set_ylabel('Count')
axarr[1].set_ylabel('Count')
axarr[2].set_ylabel('Mean')
plt.xticks(range(0, len(days)), days)
plt.xlabel('Weekday')
plt.tight_layout()
plt.show()
# Create a DataFrame grouped by state and compute the count, sum and mean
state_df = df.groupby('state').agg({'incident_id':'count', 'n_killed':['mean', sum]})
state_df.columns = ['_'.join([first, second]) for first, second in zip(state_df.columns.get_level_values(0), state_df.columns.get_level_values(1))]

# Round the data and reset the index for the plot
state_df = state_df.apply(lambda x: round(x, 3)).reset_index()

# Rename the states for a correct mapping
state_to_code = {'District of Columbia' : 'dc','Mississippi': 'MS', 'Oklahoma': 'OK', 'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI', 'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 'Nevada': 'NV', 'Maine': 'ME'}
state_df['state_code'] = state_df['state'].apply(lambda x : state_to_code[x])


# Store the data of the subplots
data = []
# Layout for the whole plot
layout = dict(title = 'Deaths grouped by state',
              width = 750,
              height = 400,
              hovermode = True)

# Count-Plot (data and layout)
data.append(dict(type = 'choropleth',
                 colorscale = [[0.0, '#baff00'],[1.0, '#000000']],
                 autocolorscale = False,
                 locations = state_df['state_code'],
                 geo = 'geo',
                 z = state_df['incident_id_count'],
                 text = state_df['state'],
                 locationmode = 'USA-states',
                 marker = dict(line = dict (color = 'rgb(255,255,255)',
                                            width = 2)),
                 colorbar = dict(title = "Incidents",
                                 x=0.29,
                                 thickness=10)))
 
layout['geo'] = dict(scope = 'usa',
                      showland=True,
                      projection=dict(type='albers usa'),
                      showlakes = True,
                      lakecolor = 'rgb(255, 255, 255)',
                      landcolor = 'rgb(229, 229, 229)',
                      subunitcolor = "rgb(255, 255, 255)",
                      domain = dict(x = [0/3,1/3], y = [0,1]))

# Sum-Plot (data and layout)
data.append(dict(type = 'choropleth',
                 colorscale = [[0.0, '#00baff'],[1.0, '#000000']],
                 autocolorscale = False,
                 locations = state_df['state_code'],
                 geo = 'geo2',
                 z = state_df['n_killed_sum'],
                 text = state_df['state'],
                 locationmode = 'USA-states',
                 marker = dict(line = dict (color = 'rgb(255,255,255)',
                                            width = 2)),
                 colorbar = dict(title = "Deaths",
                                 x=0.6225,
                                 thickness=10)))
layout['geo2'] = dict(scope = 'usa',
                      showland=True,
                      projection=dict(type='albers usa'),
                      showlakes = True,
                      lakecolor = 'rgb(255, 255, 255)',
                      landcolor = 'rgb(229, 229, 229)',
                      subunitcolor = "rgb(255, 255, 255)",
                      domain = dict(x = [1/3,2/3], y = [0,1]))

# Mean-Plot (data and layout)
data.append(dict(type = 'choropleth',
                 colorscale = [[0.0, '#4400ff'],[1.0, '#000000']],
                 autocolorscale = False,
                 locations = state_df['state_code'],
                 geo = 'geo3',
                 z = state_df['n_killed_mean'],
                 text = state_df['state'],
                 locationmode = 'USA-states',
                 marker = dict(line = dict (color = 'rgb(255,255,255)',
                                            width = 2)),
                 colorbar = dict(title = "Deaths/Incident",
                                 x=0.96,
                                 thickness=10)))
 
layout['geo3'] = dict(scope = 'usa',
                      showland=True,
                      projection=dict(type='albers usa'),
                      showlakes = True,
                      lakecolor = 'rgb(255, 255, 255)',
                      landcolor = 'rgb(229, 229, 229)',
                      subunitcolor = "rgb(255, 255, 255)",
                      domain = dict(x = [2/3,3/3], y = [0,1]))

# Create the subplots
fig = {'data':data, 'layout':layout}
iplot(fig)
# Plot the states ranked by their deadliness
state_df.set_index('state').sort_values('n_killed_mean', ascending=False)['n_killed_mean'].plot(kind='bar', figsize=(18, 4), title='Where is an incident most deadly?', grid=True, color='#4400ff')
plt.xlabel('State')
plt.ylabel('Mean deaths in an incident')
plt.xticks(rotation=75)
plt.show()
# Create a new DataFrame with the gun data
tmp = df.dropna(subset=['gun_type']).copy().reset_index()[['gun_type', 'n_killed', 'n_injured']]
# Split the guns for each incident
tmp['gun_type'] = tmp['gun_type'].apply(lambda x: [gun.replace('::', ':').split(':')[-1] for gun in x.replace('||', '|').split('|')])
# Stack the entries of the guns-list in the DataFrame
gun_series = pd.DataFrame.from_records(tmp.gun_type.tolist()).stack().reset_index(level=1, drop=True).rename('gun_type')
# Join the Data with the stacked guns
tmp = tmp.drop('gun_type', axis=1).join(gun_series).reset_index(drop=True)
gun_df = tmp.groupby('gun_type').agg({'gun_type':'count', 'n_killed':['mean', sum], 'n_injured':['mean', sum]}).apply(lambda x: round(x, 3)).drop('Unknown')
# Rename the hierarchical columns
gun_df.columns = ['_'.join([first, second]) for first, second in zip(gun_df.columns.get_level_values(0), gun_df.columns.get_level_values(1))]


# Sort and plot
gun_df['gun_type_count'].sort_values(ascending=False).plot(kind='barh', figsize=(14,5), grid=True, title='Which guns have been used in the most incidents?', color='#ff4500')
plt.xlabel('Gun Count')
plt.ylabel('Gun Type')
plt.show()
# Data for the plot
data = {'Gun_Type': gun_df.index,
        'Gun_Type_Count': gun_df.gun_type_count,
        'Size': gun_df.gun_type_count/2000+5,
        'Killed_Mean': gun_df.n_killed_mean,
        'Injured_Mean': gun_df.n_injured_mean}
source = ColumnDataSource(data=data)

# Settings to display the data by hovering
hover = HoverTool(tooltips=[
    ('Gun Type', '@Gun_Type'),
    ('Incident Count', '@Gun_Type_Count'),
    ('Killed Mean', '@Killed_Mean'),
    ('Injured Mean', '@Injured_Mean')])
wheel = WheelZoomTool()

# Create the plot
p = figure(plot_width=750, plot_height=400, title='Mean killed/injured in an incident grouped by gun and scaled by incident-count', tools=[hover, wheel, ResetTool(), PanTool()], match_aspect=True)
p.toolbar.active_scroll = wheel
p.circle(x='Injured_Mean', y='Killed_Mean', size='Size', source=source, color='#4400ff')
p.text(x='Injured_Mean', y='Killed_Mean', text='Gun_Type', text_font_size='10pt', source=source)
p.xaxis.axis_label = 'Mean: Injured'
p.yaxis.axis_label = 'Mean: Killed'
show(p)
# Number of cities
n = 100

city_df = df.groupby('city_or_county').agg({'n_killed':['mean', sum, 'count'], 'longitude':'mean', 'latitude':'mean'})
city_df.columns = ['_'.join([first, second]) for first, second in zip(city_df.columns.get_level_values(0), city_df.columns.get_level_values(1))]
#tmp[tmp.n_killed_count>50].sort_values('n_killed_mean', ascending=False).head(10)
city_df = city_df[city_df.n_killed_count>50].sort_values('n_killed_sum').tail(n).reset_index()

# Death colorscale
scl = [[0.0, '#00baff'],[1.0, '#000000']]

# Data for the map
data = [ dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = city_df['longitude_mean'],
        lat = city_df['latitude_mean'],
        text = city_df['city_or_county'] +': '+ city_df['n_killed_sum'].astype(str),
        mode = 'markers',
        marker = dict(
            size = 8,
            autocolorscale = False,
            symbol = 'square',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
            color = city_df['n_killed_sum'],
            cmax = city_df['n_killed_sum'].max(),
            colorbar=dict(
                title="Deaths"
            )
        ))]

# Layout for the map
layout = dict(
        title = '{} Cities with the most deaths in 4.25 years'.format(n),
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(0, 0, 0)",
            countrycolor = "rgb(0, 0, 0)",
            countrywidth = 0.5,
            subunitwidth = 0.5))

# Create the map
fig = dict(data=data, layout=layout)
iplot(fig)
# Store the data
data = []

# Iterate over each row
for row in df.values:
    # Get the data
    length = []
    state = row[1]
    n_killed = row[4]
    n_injured = row[5]
    participant_age = row[18]
    
    # Split the entries in the columns
    if type(participant_age)==str:
        participant_age = participant_age.replace('||', '|').split('|')
        length.append(len(participant_age))
    participant_gender = row[20]
    if type(participant_gender)==str:
        participant_gender = participant_gender.replace('||', '|').split('|')
        length.append(len(participant_gender))
    participant_status = row[23]
    if type(participant_status)==str:
        participant_status = participant_status.replace('||', '|').split('|')
        length.append(len(participant_status))
    participant_type = row[24]
    if type(participant_type)==str:
        participant_type = participant_type.replace('||', '|').split('|')
        length.append(len(participant_type))
        
    # Combine the splitted entries
    if length:
        for i in range(max(length)):
            try:
                p_a = participant_age[i].replace('::', ':').split(':')[-1]
            except:
                p_a = np.nan
            try:
                p_g = participant_gender[i].replace('::', ':').split(':')[-1]
            except:
                p_g = np.nan
            try:
                p_s = participant_status[i].replace('::', ':').split(':')[-1]
            except:
                p_s = np.nan
            try:
                p_t = participant_type[i].replace('::', ':').split(':')[-1]
            except:
                p_t = np.nan
                
            # Store the data
            data.append([state, n_killed, n_injured, p_a, p_g, p_s, p_t])

# Create the DataFrame
people_df = pd.DataFrame(data, columns=['state', 'n_killed', 'n_injured', 'age', 'gender', 'status', 'type'])    
people_df['age'] = people_df['age'].astype(float)

# Groupby categories, compute mean
#participant_df = people_df.groupby(['gender', 'status', 'type']).agg({'n_killed':['mean', sum], 'n_injured':['mean', sum], 'age':'mean', 'state':'count'})
#participant_df.columns = ['_'.join([first, second]) for first, second in zip(participant_df.columns.get_level_values(0), participant_df.columns.get_level_values(1))]


# Color palette
palette =["#ff4500","#00baff"]

people_df[(people_df.type=='Victim') & (people_df.gender.isin(['Male', 'Female']))].groupby('gender').state.count().plot(kind='bar', color=palette, title='Involved victims grouped by gender')
plt.ylabel('Victims')
plt.show()
# Color palette
palette ={"Female":"#ff4500","Male":"#00baff"}

# Create plots
f, axarr = plt.subplots(1, 2, sharey=True, figsize=(14,6))
sns.barplot(data=people_df[(people_df.gender.isin(['Male', 'Female'])) & (people_df.type=='Subject-Suspect')], x='gender', y='n_injured', ax=axarr[0], palette=palette)
sns.barplot(data=people_df[(people_df.gender.isin(['Male', 'Female'])) & (people_df.type=='Subject-Suspect')], x='gender', y='n_killed', ax=axarr[1], palette=palette)
axarr[0].set_title('Mean injured grouped by attacker gender')
axarr[1].set_title('Mean killed grouped by attacker gender')
axarr[0].set_ylabel('mean injured')
axarr[1].set_ylabel('mean killed')
plt.show()
# Create plots
sns.barplot(data=people_df[(people_df.gender.isin(['Male', 'Female'])) & (people_df.type=='Subject-Suspect')], x='gender', y='age', palette=palette)
plt.title('Attacker age grouped by gender')
plt.show()
age_df = people_df[people_df.type=='Subject-Suspect'].groupby('state').agg({'age':'mean'}).apply(lambda x: round(x, 2)).reset_index()
age_df['state_code'] = age_df['state'].apply(lambda x : state_to_code[x])

scl = [[0.0, '#ff4500'],[1.0, '#000000']]

# Data for the map
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = age_df['state_code'],
        z = age_df['age'].astype(float),
        locationmode = 'USA-states',
        text = age_df['state'],
        marker = dict(line = dict (color = 'rgb(255,255,255)',
                      width = 2)),
        colorbar = dict(title = "Age"))]

# Layout for the map
layout = dict(title = 'Attacker age grouped by state',
              geo = dict(
              scope='usa',
              projection=dict( type='albers usa' ),
              showlakes = True,
              lakecolor = 'rgb(255, 255, 255)'))
    
fig = dict(data=data, layout=layout)
iplot(fig)
frequencies = df['location_description'].value_counts().to_dict()

wordcloud = WordCloud(height=600, width=800, background_color='white')
wordcloud.generate_from_frequencies(frequencies=frequencies)
plt.figure(figsize=(14,6))
plt.title('Count Of Locations')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
