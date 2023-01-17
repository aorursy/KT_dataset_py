# Import the packages we'll need to read in and process the data

import numpy as np # Linear algebra and matrix operations
import pandas as pd # Data processing
import seaborn as sns # Plotting
import matplotlib.pyplot as plt # More plotting

buildings = pd.read_csv('../input/seattle-unreinforced-masonry-buildings/unreinforced-masonry-buildings.csv')
# Print the first few rows to see what our data looks like.

buildings.head()
# Print some basic stats about the data.

buildings.describe()

# 1144 Buildings
# Mean building year built: 1900 -- Note: minimum age is set to 0, so we'll have to deal with that data.
# Mean # stories: 2.5 (max 10)
# First, replace Year Built of 0 for NaN, since it's not a valid number.

buildings['Year Built'] = buildings['Year Built'].replace({0:np.nan})

# Let's see if there's any missing data.

buildings.isnull().sum()

# A few data points are missing: Year Built (10), Estimated Number of Occupants (2) and Confirmation Source (5)

# Okay, now let's look at each feature separately, starting with building age.

# For now, we'll drop rows with missing data, since it's a small minority of the overall dataset.

complete_buildings = buildings.dropna()

# Next, plot the year built.
sns.distplot(complete_buildings['Year Built'])

# Looks like most of the unreinforced masonry buildings were built in ~1910 and ~1930
# Now let's look at the number of stories

sns.countplot(complete_buildings['No. Stories'])
# Let's look at the Risk category

sns.countplot(complete_buildings['Preliminary Risk Category'])

# Most are medium risk, critical risk is a small percentage.
# Let's also see which neighborhoods are represented.

complete_buildings['Neighborhood'].value_counts()

# Where are the high risk buildings located?

critical_risk_buildings = complete_buildings[complete_buildings['Preliminary Risk Category'] == 'Critical Risk']
critical_risk_buildings['Neighborhood'].value_counts()

# Lastly, let's look at the building type.

complete_buildings['Building Use'].value_counts()

# 123 Public Assembly and 73 schools! Oh shit.
# Cool, now let's start exploring the relationships between different variables.

# First let's plot age and risk level.

sns.stripplot(x="Preliminary Risk Category", y="Year Built", data=complete_buildings, jitter=True)

# Interestingly, it doesn't appear that the age of the building has much effect on the risk level.
# Old buildings and newer buildings are distributed fairly comparably

# What about number of stories and risk level?

sns.stripplot(x="Preliminary Risk Category", y="No. Stories", data=complete_buildings, jitter=True)

# Also doesn't show a clean delineation... 

# Just for fun, let's see what the relationship is between Year Built and Number of stories

sns.regplot(x='Year Built', y='No. Stories', data=complete_buildings)

# Sort of surprising. Building height went down over time for buildings at risk. That might
# be because the taller buildings remained (and are at risk now.)
# We have the addresses for each building. Let's look into Geocoding that data.

# Python doesn't seem to have a great batch geocoding package, since it relies heavily
# on third-party web services. So, we'll export the address data to Google sheets.

# complete_buildings['Address'].to_csv('address.csv')


lat_long = pd.read_csv('../input/masonry-buildings-latlong/Untitled spreadsheet - Sheet1.csv')


# We only ahve 998 lat/long data points due to an API restriction
abbreviated_buildings = complete_buildings.head(995)
abbreviated_buildings['Lat'] = lat_long.iloc[:,0]
abbreviated_buildings['Long'] = lat_long.iloc[:,1]

abbreviated_buildings = abbreviated_buildings.head(995).dropna()
abbreviated_buildings.describe()


import folium

map = folium.Map(location=[abbreviated_buildings['Lat'].mean(),abbreviated_buildings['Long'].mean()], zoom_start=11, tiles="Stamen Toner")

def color(risk):
    if risk == 'Critical Risk':
        col='red'
    elif risk == 'High Risk':
        col='yellow'
    elif risk == 'Medium Risk':
        col='green'
    return col

print(abbreviated_buildings.describe())

for lat, long, risk in zip(abbreviated_buildings['Lat'], abbreviated_buildings['Long'], abbreviated_buildings['Preliminary Risk Category']):
    map.add_child(folium.Marker(location=[lat,long], icon=folium.Icon(color=color(risk),icon_color='green')))

map
map.save(outfile='map.html')
