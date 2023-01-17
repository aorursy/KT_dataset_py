from IPython.display import Image
Image("../input/kernelassets/pexels-rosemary-ketchum-1464230.jpg")
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pandas_profiling import ProfileReport

import warnings

from tqdm.notebook import tqdm



from plotly import express as px

from plotly import graph_objs as go

from plotly import figure_factory as ff



from colorama import Fore, Style



from geopy.geocoders import Nominatim



warnings.simplefilter("ignore")

plt.style.use("fivethirtyeight")

geolocator = Nominatim(user_agent="police-shooting-viz")
def cout(string: str, color=Fore.RED):

    """

    Utility function to string in colors

    """

    print(color+string+Style.RESET_ALL)
def statistics(dataframe, column):

    cout(f"The Average value in {column} is: {dataframe[column].mean():.2f}", Fore.RED)

    cout(f"The Standard Deviation of {column} is: {dataframe[column].std():.2f}", Fore.LIGHTCYAN_EX)

    cout(f"The Maximum value in {column} is: {dataframe[column].max()}", Fore.BLUE)

    cout(f"The Minimum value in {column} is: {dataframe[column].min()}", Fore.YELLOW)

    cout(f"The 25th Quantile of {column} is: {dataframe[column].quantile(0.25)}", Fore.GREEN)

    cout(f"The 50th Quantile of {column} is: {dataframe[column].quantile(0.50)}", Fore.CYAN)

    cout(f"The 75th Quantile of {column} is: {dataframe[column].quantile(0.75)}", Fore.MAGENTA)
# Let's read the data

data = pd.read_csv("../input/us-police-shootings/shootings.csv")

data.head()
# Get Date and Month from the Data and store it in a seperate column

all_year = []

all_months = []



months = {

    '01':'January',

    '02':'February',

    '03':'March',

    '04':'April',

    '05':'May',

    '06':'June',

    '07':'July',

    '08':'August',

    '09':'September',

    '10':'October',

    '11':'November',

    '12':'December'

}



def get_date(datetime):

    date = str(datetime)

    year = datetime[:4]

    month = datetime[5:7]

    return year, months[month]



for date in data['date']:

    yr, mn = get_date(date)

    all_year.append(yr)

    all_months.append(mn)



data['year'] = all_year

data['month'] = all_months
# Year BarPlot

targets = data['year'].value_counts().tolist()

values = list(dict(data['year'].value_counts()).keys())



fig = px.bar(

    x=values,

    y=targets,

    color=values,

    labels={'x':'Years', 'y':'Number of Incidents'},

    title="Number of Incidents over years"

)



fig.show()
# Months BarPlot

targets = data['month'].value_counts().tolist()

values = list(dict(data['month'].value_counts()).keys())



fig = px.bar(

    x=values,

    y=targets,

    color=values,

    labels={'x':'Months', 'y':'Number of Incidents'},

    title="Number of Incidents over months"

)



fig.show()
# Pie Chart to see the manner of death

targets = data['manner_of_death'].value_counts().tolist()

values = list(dict(data['manner_of_death'].value_counts()).keys())



fig = px.pie(

    values=targets, 

    names=values,

    title='Manner of Death in Police Shootings',

    color_discrete_sequence=['gray', 'black']

)

fig.show()
# Months BarPlot

targets = data['armed'].value_counts().tolist()

values = list(dict(data['armed'].value_counts()).keys())



fig = px.bar(

    x=values,

    y=targets,

    color=values,

    labels={'x':'Armament Type', 'y':'Number of Incidents'},

    title="Number of Incidents with different armaments"

)



fig.show()
statistics(data, "age")
# Let's plot the age column too

plt.style.use("classic")

sns.distplot(data['age'], color='blue')

plt.title(f"Age Distribution [\u03BC : {data['age'].mean():.2f} years | \u03C3 : {data['age'].std():.2f} years]")

plt.xlabel("Age")

plt.ylabel("Count")

plt.show()
plt.style.use("fivethirtyeight")

plt.figure(figsize=(16, 6))

sns.kdeplot(data.loc[data['gender'] == 'M', 'age'], label = 'Male',shade=True)

sns.kdeplot(data.loc[data['gender'] == 'F', 'age'], label = 'Female',shade=True)



# Labeling of plot

plt.xlabel('Age')

plt.ylabel('Density')

plt.title('Distribution of Ages for Male and Female Individuals')

plt.show()
# Pie Chart to see gender of individuals

targets = data['gender'].value_counts().tolist()

values = list(dict(data['gender'].value_counts()).keys())



fig = px.pie(

    values=targets, 

    names=values,

    title='Gender of Individuals in Police Shootings',

    color_discrete_sequence=['blue', 'magenta']

)

fig.show()
targets = data[data['gender']=='F']['armed'].value_counts().tolist()

values = list(dict(data[data['gender']=='F']['armed'].value_counts()).keys())



fig = px.pie(

    values=targets, 

    names=values,

    title='Armaments held by Women during Shootings',

)

fig.show()
# Pie Chart to see the manner of death

targets = data['race'].value_counts().tolist()

values = list(dict(data['race'].value_counts()).keys())



fig = px.pie(

    values=targets,

    names=values,

    title='Races of individuals in Police Shootings',

)



fig.show()
# First, let's make a new dataframe with only cities arrange according to number of incidents uniquely

city_names = dict(data['city'].value_counts()).keys()

city_incidents = data['city'].value_counts().tolist()



city_df = pd.DataFrame()

city_df['name'] = city_names

city_df['incidents'] = city_incidents

city_df.head()
# Let's add the corresponding longitude and latitude to the cities.

longs, lats = [], []

err_idx = []

for idx, city in tqdm(enumerate(city_df['name'])):

    loc = geolocator.geocode(city)

    try:

        longs.append(loc.longitude)

        lats.append(loc.latitude)

    except:

        err_idx.append(idx)

        

city_df = city_df.drop(err_idx)

city_df['lat'] = lats

city_df['lon'] = longs
# Draw a bubble map for city of incidents.

city_df['text'] = city_df['name'] + '<br>Incidents: ' + (city_df['incidents']).astype(str)

limits = [(0, 4), (5, 12), (12, 22), (22, 33), (33, 43)]

colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]

cities = []



fig = go.Figure()



for i in range(len(limits)):

    lim = limits[i]

    df_sub = city_df[lim[0]:lim[1]]

    fig.add_trace(go.Scattergeo(

        locationmode = 'USA-states',

        lon = df_sub['lon'],

        lat = df_sub['lat'],

        text = df_sub['text'],

        marker = dict(

            size = df_sub['incidents'],

            color = colors[i],

            line_color='rgb(40,40,40)',

            line_width=0.5,

            sizemode = 'area'

        ),

        name = '{0} - {1}'.format(lim[0],lim[1])))



fig.update_layout(

        title_text = 'Police Shooting Incident across US Cities<br>(Click legend to toggle traces)',

        showlegend = True,

        geo = dict(

            scope = 'usa',

            landcolor = 'rgb(200, 200, 200)',

        )

    )



fig.show()
# Just like the cities, we will make a new dataframe

state_names = dict(data['state'].value_counts()).keys()

state_incidents = data['state'].value_counts().tolist()



state_df = pd.DataFrame()

state_df['state'] = state_names

state_df['incidents'] = state_incidents

state_df.head()
# Choropleth map

fig = go.Figure(data=go.Choropleth(

    locations=state_df['state'], # Spatial coordinates

    z = state_df['incidents'].astype(int), # Data to be color-coded

    locationmode = 'USA-states', # set of locations

    colorscale = 'amp', # color scale

    colorbar_title = "Incidents Density", # title for the color bar

))



fig.update_layout(

    title_text = 'Police Shooting Incidents Across US States', # title for the plot

    geo_scope='usa', # limite map scope to USA

)



fig.show()
# Pie Chart to see the manner of death

targets = data['signs_of_mental_illness'].value_counts().tolist()

values = list(dict(data['signs_of_mental_illness'].value_counts()).keys())



fig = px.pie(

    values=targets,

    names=values,

    title='Signs of Mental Illness in individuals',

    color_discrete_sequence=['blue', 'red']

)



fig.show()
# Pie Chart to see the manner of death

targets = data['threat_level'].value_counts().tolist()

values = list(dict(data['threat_level'].value_counts()).keys())



fig = px.pie(

    values=targets,

    names=values,

    title='Threat Level Posed by Individuals',

    color_discrete_sequence=['orange', 'yellow', 'gold']

)



fig.show()
# Pie Chart to see the manner of death

targets = data['flee'].value_counts().tolist()

values = list(dict(data['flee'].value_counts()).keys())



fig = px.pie(

    values=targets,

    names=values,

    title='How Individuals were Fleeing the Crime Scene',

)



fig.show()
# Pie Chart to see the manner of death

targets = data['body_camera'].value_counts().tolist()

values = list(dict(data['body_camera'].value_counts()).keys())



fig = px.pie(

    values=targets,

    names=values,

    title='Possession of Body Camera',

)



fig.show()
# Months BarPlot

targets = data['arms_category'].value_counts().tolist()

values = list(dict(data['arms_category'].value_counts()).keys())



fig = px.bar(

    x=values,

    y=targets,

    color=values,

    labels={'x':'Armament Type', 'y':'Number of Incidents'},

    title="Number of Incidents with different armaments"

)



fig.show()
profile = ProfileReport(data, title="Pandas Profiling Report")
profile.to_notebook_iframe()