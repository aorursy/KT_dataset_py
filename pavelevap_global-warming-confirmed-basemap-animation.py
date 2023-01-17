import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



# using Basemap for map visualization. Installed it with "conda install basemap"

from mpl_toolkits.basemap import Basemap

from matplotlib import animation, rc

from IPython.display import HTML



import warnings

warnings.filterwarnings('ignore')
globaltemp = pd.read_csv('../input/GlobalTemperatures.csv', parse_dates=['dt'])

globaltemp.head(3)
year_temp = globaltemp.groupby(globaltemp.dt.dt.year).mean()

pd.stats.moments.ewma(year_temp.LandAverageTemperature, 5).plot()

year_temp.LandAverageTemperature.plot(linewidth=1)

plt.title('Average temperature by year')

plt.xlabel('year')
bycities = pd.read_csv('../input/GlobalLandTemperaturesByCity.csv', parse_dates=['dt'])

# there are some cities with the same name but in different countries 

bycities[['City', 'Country']].drop_duplicates()

bycities.City = bycities.City.str.cat(bycities.Country, sep=' ')

bycities = bycities[bycities.dt.dt.year >= 1900]

bycities.head()
city_means = bycities.groupby(['City', bycities.dt.dt.year])['AverageTemperature'].mean().unstack()

city_mins = bycities.groupby(['City', bycities.dt.dt.year])['AverageTemperature'].min().unstack()

city_maxs = bycities.groupby(['City', bycities.dt.dt.year])['AverageTemperature'].max().unstack()

city_means.head()
first_years_mean = city_means.iloc[:, :5].mean(axis=1) # mean temperature for the first 5 years

city_means_shifted = city_means.subtract(first_years_mean, axis=0)



def plot_temps(cities, city_ser, ax):

    first_years_mean = city_ser.iloc[:, :5].mean(axis=1)

    city_ser = city_ser.subtract(first_years_mean, axis=0)

    for city in random_cities:

        row = city_ser.loc[city]

        pd.stats.moments.ewma(row, 10).plot(label=row.name, ax=ax)

    ax.set_xlabel('')

    ax.legend(loc='best')



fig, axes = plt.subplots(3,1, figsize=(10,10))



n = 5

random_cities = city_means_shifted.sample(n).index



plot_temps(random_cities, city_means, axes[0])

plot_temps(random_cities, city_mins, axes[1])

plot_temps(random_cities, city_maxs, axes[2])



axes[0].set_title("Year's mean temperature increase for random cities")

axes[1].set_title("Year's min temperature increase for random cities")

axes[2].set_title("Year's max temperature increase for random cities")
cities_info = bycities.groupby(['City']).first()

cities_info.head()
def get_temp_markers(city_names, year):

    points = np.zeros(len(city_names), dtype=[('lon', float, 1),

                                      ('lat', float, 1),

                                      ('size',     float, 1),

                                      ('color',    float, 1)])

    cmap = plt.get_cmap('coolwarm')

    

    for i, city in enumerate(city_names):

        city_temps = city_means.loc[city]

        _MIN, _MAX, _MEDIAN = city_temps.min(), city_temps.max(), city_temps.median()

        temp = city_temps.loc[year]

        

        coords = cities_info.loc[city][['Latitude', 'Longitude']].values

        lat = float(coords[0][:-1]) * (-1 if coords[0][-1] == 'S' else 1)

        lon = float(coords[1][:-1]) * (-1 if coords[1][-1] == 'W' else 1)

        

        points['lat'][i] = lat

        points['lon'][i] = lon

        points['size'][i] = 100 * abs(temp - _MEDIAN)

        points['color'][i] = (temp - _MIN) / (_MAX - _MIN)

            

    return points   
fig = plt.figure(figsize=(10, 10))

cmap = plt.get_cmap('coolwarm')



map = Basemap(projection='cyl')

map.drawmapboundary()

map.fillcontinents(color='lightgray', zorder=1)



START_YEAR = 1950

LAST_YEAR = 2013



n_cities = 500

random_cities = city_means.sample(n_cities).index

year_text = plt.text(-170, 80, str(START_YEAR),fontsize=15)



temp_markers = get_temp_markers(random_cities, START_YEAR)

xs, ys = map(temp_markers['lon'], temp_markers['lat'])

scat = map.scatter(xs, ys, s=temp_markers['size'], c=temp_markers['color'], cmap=cmap, marker='o', 

                   alpha=0.3, zorder=10)



def update(frame_number):

    current_year = START_YEAR + (frame_number % (LAST_YEAR - START_YEAR + 1))

    

    temp_markers = get_temp_markers(random_cities, current_year)

    xs, ys = map(temp_markers['lon'], temp_markers['lat'])



    scat.set_offsets(np.dstack((xs, ys)))

    scat.set_color(cmap(temp_markers['color']))

    scat.set_sizes(temp_markers['size'])

    

    year_text.set_text(str(current_year))



# # # Construct the animation, using the update function as the animation

# # # director.

ani = animation.FuncAnimation(fig, update, interval=500, frames=LAST_YEAR - START_YEAR + 1)



cbar = map.colorbar(scat, location='bottom')

cbar.set_label('0.0 -- min temperature record for the city   1.0 -- max temperature record for the city')

plt.title('Mean year temperatures for {} random cities'.format(n_cities))

plt.show()
ani.save('animation.gif', writer='imagemagick', fps=2)
import io

import base64



filename = 'animation.gif'



video = io.open(filename, 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))