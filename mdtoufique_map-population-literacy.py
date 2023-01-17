import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
from numpy import array
import seaborn  as sns
cities = pd.read_csv('../input/cities_r2.csv')
cities.head()
cities.describe()
#cities.plot(x="state_name",y=["population_total"],kind= "bar",figsize=(100,100))
popByState=cities.groupby(["state_name"],axis=0,as_index=False).sum()
popByState.head()
#popByState.describe()
popByState.plot(x="state_name",y=["population_total","population_male","population_female"],kind="barh",figsize=(20,10))
popByState.plot(x="state_name",y=["literates_total","literates_male","literates_female"],kind="barh",figsize=(20,10))
popByState["illiterates_total"]=popByState["population_total"]-popByState["literates_total"]
popByState["illiterates_male"]=popByState["population_male"]-popByState["literates_male"]
popByState["illiterates_female"]=popByState["population_female"]-popByState["literates_female"]
popByState.plot(x="state_name",y=["illiterates_total","illiterates_male","illiterates_female"],kind="barh",figsize=(20,10))
sns.distplot(popByState)
cities["lat"] = cities['location'].apply(lambda x: float(x.split(',')[0]))
cities["long"] = cities['location'].apply(lambda x: float(x.split(',')[1]))
def plot_map(sizes, colorbarValue):

    f, ax = plt.subplots(figsize=(12, 9))

    # initialize Basemap
    map = Basemap(width=5000000,
                  height=3500000,
                  resolution='l',
                  projection='aea',
                  llcrnrlon=69,
                  llcrnrlat=6,
                  urcrnrlon=99,
                  urcrnrlat=36,
                  lon_0=78,
                  lat_0=20,
                  ax=ax)

    # draw map boundaries
    map.drawmapboundary(fill_color='white')
    map.fillcontinents(color='#313438', lake_color='#313438', zorder=0.5)
    map.drawcountries(color='white')

    # show scatter point on map
    x, y = map(array(cities["long"]), array(cities["lat"]))
    cs = map.scatter(x, y, s=sizes, marker="o", c=sizes, cmap=cm.Dark2, alpha=0.5)

    # add colorbar.
    cbar = map.colorbar(cs, location='right',pad="5%")
    cbar.ax.set_yticklabels(colorbarValue)

    plt.show()
population_sizes = cities["population_total"].apply(lambda x: int(x / 5000))
colorbarValue = np.linspace(cities["population_total"].min(), cities["population_total"].max(), 
                            num=10)
colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)
literacy_sizes = cities["literates_total"].apply(lambda x: int(x / 2000))
colorbarValue = np.linspace(cities["literates_total"].min(), cities["literates_total"].max(), 
                            num=10)
colorbarValue = colorbarValue.astype(int)

plot_map(literacy_sizes, colorbarValue)
cities.columns
plt.bar(cities["name_of_city"],cities["literates_total"],width=5)
