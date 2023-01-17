import psycopg2
import pandas as pd 

connection = psycopg2.connect(
    host="localhost",
    database="OpenFlights",
    user="postgres",
    password="postgresGh")

#Read in airports data
airports=pd.read_sql("SELECT * FROM airports", connection)
#Read in airlines data
airlines=pd.read_sql("SELECT * FROM airlines", connection)
#Read in routes data
routes=pd.read_sql("SELECT * FROM routes", connection)

airports.head()
airlines.head()
routes.head()
# The following data cleaning ensures that we have only numeric data in the airline_id column
routes = routes[routes["airlineid"] != "\\N"]

# 
# Make a histogram
# 
import math
def haversine(lon1, lat1, lon2, lat2):
    # Convert coordinates to floats.
    lon1, lat1, lon2, lat2 = [float(lon1), float(lat1), float(lon2), float(lat2)]
    # Convert to radians from degrees.
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # Compute distance.
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    km = 6367 * c
    return km

# Function that calculates distance between the source and dest airports for a single route
def calc_dist(row):
    dist = 0
    try:
        # Match source and destination to get coordinates.
        source = airports[airports["airportid"] == row["srcid"]].iloc[0]
        dest = airports[airports["airportid"] == row["destid"]].iloc[0]
        # Use coordinates to compute distance.
        dist = haversine(dest["longitude"], dest["latitude"], source["longitude"], source["latitude"])
    except (ValueError, IndexError):
        pass
    return dist
# Apply the distance calculation function across the routes dataframe,
# getting a pandas series containing all the route length (in kilometers)
route_lengths = routes.apply(calc_dist, axis=1)
route_lengths.head()
# Create a histogram, which will bin the values into ranges and count how many routes fall into each range
import matplotlib.pyplot as plt
%matplotlib inline
plt.hist(route_lengths, bins=20)
import holoviews as hv
from holoviews import opts, Cycle
hv.extension('bokeh')
hv.Distribution(route_lengths)
#Convert route_lengths series into a dataframe
route_lengths_df = route_lengths.to_frame()
route_lengths_df.head()
import numpy as np
from bokeh.plotting import figure
from bokeh.io import show, output_notebook

output_notebook()
x,y = np.histogram(route_lengths_df[0],bins=20)
histogram_df = pd.DataFrame({ '0': x, 'left': y[:-1], 'right':y[1:]})
p = figure(plot_height = 400, plot_width = 400, 
           title = 'Histogram of Route Lengths Distribution',
          )
# Add a quad glyph
p.quad(bottom=0, top=histogram_df['0'], 
       left=histogram_df['left'], right=histogram_df['right'], 
       fill_color='blue', line_color='black')

# Show the plot
show(p)