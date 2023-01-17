#checking the shape

import pandas as pd

deaths = pd.read_csv("../input/ghost-map/deaths.csv")

deaths.shape
deaths.head()
deaths.info()
#describing the dataframe

deaths.describe()
# creating pair of longitude and lattitude

locations = deaths[["X coordinate","Y coordinate"]]

deaths_list = locations.values.tolist()

len(deaths_list)
#Plotting map

import folium

map = folium.Map(location=[51.5132119,-0.13666], tiles='Stamen Toner', zoom_start=17)

for point in range(0, len(deaths_list)):

    folium.CircleMarker(deaths_list[point], radius=8, color='red', fill=True, fill_color='red', opacity = 0.4).add_to(map)

map
# getting the co-ordinate pairs for map

pumps = pd.read_csv('../input/ghost-map/pumps.csv')

locations_pumps = pumps.loc[:, ['X coordinate','Y coordinate']]

pumps_list = locations_pumps.values.tolist()



# Plotting the map

map1 = map

for point in range(0, len(pumps_list)):

    folium.Marker(deaths_list[point], popup=pumps['Pump Name'][point]).add_to(map1)

map1
dates = pd.read_csv('../input/ghost-map/dates.csv', parse_dates=['date'])



# Set the Date when handle was removed (8th of September 1854)

handle_removed = pd.to_datetime('1854/9/8')

dates['day_name'] = dates['date'].dt.day_name()



# Creating new column "handle" in "dates" DataFrame based on a Date the handle was removed 

dates['handle'] = dates['date'] > handle_removed

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