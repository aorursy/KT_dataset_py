# Loading in the pandas module
# ... YOUR CODE FOR TASK 1 ...
import pandas as pd
import numpy as np

# Reading in the data
deaths = pd.read_csv("../input/dataset/deaths.csv")

# Print out the shape of the dataset
deaths.shape

# Printing out the first 5 rows
deaths.head(5)
# Summarizing the content of deaths
# ... YOUR CODE FOR TASK 2 ...
deaths.info()

# Define the new names of your columns
newcols = {
    'Death': 'death_count', 
    'Y coordinate': 'y_longitude',
   'X coordinate':'x_latitude'
    
    
    }

# Rename your columns
deaths.rename(columns=newcols, inplace=True)

# Describe the dataset
deaths.describe()
# Create `locations` by subsetting only Latitude and Longitude from the dataset 
locations = deaths[["x_latitude","y_longitude"]]

# Create `deaths_list` by transforming the DataFrame to list of lists 
deaths_list = locations.values.tolist()

# Check the length of the list
# ... YOUR CODE FOR TASK 3 ...
len(deaths_list)
# Plot the data on map (map location is provided) using folium and for loop for plotting all the points
import folium

map = folium.Map(location=[51.5132119,-0.13666], tiles='Stamen Toner', zoom_start=17)
for point in range(0, len(deaths_list)):
    folium.CircleMarker(deaths_list[point], radius=8, color='red', fill=True, fill_color='red', opacity = 0.4).add_to(map)
map
# Import the data
pumps = pd.read_csv("../input/dataset/pumps.csv")

# Subset the DataFrame and select just ['X coordinate', 'Y coordinate'] columns
locations_pumps = pumps[["X coordinate","Y coordinate"]]

# Transform the DataFrame to list of lists in form of ['X coordinate', 'Y coordinate'] pairs
pumps_list = locations_pumps.values.tolist()

# Create a for loop and plot the data using folium (use previous map + add another layer)
map1 = map
for point in range(0, len(pumps_list)):
    folium.Marker(pumps_list[point], popup=pumps['Pump Name'][point]).add_to(map1)
map1
# Import the data the right way
dates = pd.read_csv('../input/dataset/dates.csv', parse_dates=['date'])

# Set the Date when handle was removed (8th of September 1854)
handle_removed = pd.to_datetime('1854/9/8')

# Create new column `day_name` in `dates` DataFrame with names of the day 
dates['day_name'] = dates.date.dt.day_name

# Create new column `handle` in `dates` DataFrame based on a Date the handle was removed 
dates['handle'] = dates.date > handle_removed

# Check the dataset and datatypes
dates.info()

# Create a comparison of how many cholera deaths and attacks there were before and after the handle was removed
dates.groupby(['handle']).sum()
import bokeh
from bokeh.plotting import output_notebook, figure, show
output_notebook(bokeh.resources.INLINE)

# Set up figure
p = figure(plot_width=900, plot_height=450, x_axis_type='datetime', tools='lasso_select, box_zoom, save, reset, wheel_zoom',
          toolbar_location='above', x_axis_label='Date', y_axis_label='Number of Deaths/Attacks', 
          title='Number of Cholera Deaths/Attacks before and after 8th of September 1854 (removing the pump handle)')

# Plot on figure
p.line(dates['date'], dates['deaths'], color='red', alpha=1, line_width=3, legend='Cholera Deaths')
p.circle(dates['date'], dates['deaths'], color='black', nonselection_fill_alpha=0.2, nonselection_fill_color='grey')
p.line(dates['date'], dates['attacks'], color='black', alpha=1, line_width=2, legend='Cholera Attacks')

show(p)
# Based on John Snow's map and the data John Snow collected, what would you say?
john_snow_knows_nothing = False