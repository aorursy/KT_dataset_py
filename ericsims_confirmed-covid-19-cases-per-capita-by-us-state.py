# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import folium
# Thank you to MaxU on StackOverflow for this suggestion to get around 403 errors!



import requests



url = 'https://www.worldometers.info/coronavirus/country/us/'



header = {

  "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",

  "X-Requested-With": "XMLHttpRequest"

}



r = requests.get(url, headers=header)



dfs = pd.read_html(r.text)
cv = dfs[0]

cv.head()
pop_den = pd.read_csv("../input/population-density-by-state-us-2020/pop_den.csv")

pop_den.sort_values("Population", ascending=False, inplace=True)

pop_den.head()
pop_den = pop_den.set_index("State", drop=True)
df = cv.join(pop_den, on="USAState")

df.head()
us_lat_long = pd.read_csv("../input/usa-latlong-for-state-abbreviations/statelatlong.csv")

us_lat_long.drop("State", axis=1, inplace=True)

# us_lat_long = us_lat_long.rename(index={"City": "State"})

us_lat_long = us_lat_long.set_index("City", drop=True)

us_lat_long.head()
df = df.join(us_lat_long, on="USAState")

df.head()
#Calculate cases per capita

df["CasesPerCapita"] = df["TotalCases"]/df["Population"]



#Remove Source column

df.drop(["Source"], axis=1, inplace=True)



#Sort by CasesPerCapita column

df.sort_values("CasesPerCapita", ascending=False, inplace=True)



#Drop Rows with NaN Latitude (just needed to choose a location point)

df.dropna(subset=['Latitude'], inplace=True)



#Convert TotalCases to float64 (for choropleth markers)

df = df.astype({'TotalCases': 'float64'})



df.head()
fig, ax = plt.subplots(figsize=(15,5))



a1_label = "Cases per Capita"

a2_label = "State Population"



plt.xticks(rotation=90)

ax2=ax.twinx()

ax2.set_ylabel("State Population in Tens of Millions", fontsize=14)

# ax2.legend(loc="upper right")



ax.legend(loc="upper left")

ax.set_title("Cases per Capita and State Population", fontsize=20)

bar1 = ax.bar('USAState', 'CasesPerCapita', data=df, width=-.3, align='edge', label="Cases per Capita")

ax.set_ylabel("Cases per Capita", fontsize=14)

# ax.set_facecolor('xkcd:charcoal')

# ax.set_yscale(.005)

bar2 = ax2.bar('USAState', 'Population', data=df, color="pink", width=.2, align='edge', label="State Population")





from matplotlib.patches import Rectangle



p1 = Rectangle((0, 0), 1, 1)

p2 = Rectangle((0, 0), 1, 1, fc="pink")

ax.legend([p1, p2], [a1_label, a2_label], loc="upper right")



fig.tight_layout()



plt.savefig("CasesPerCapita.png")
fig, ax = plt.subplots(figsize=(15,5))



a1_label = "Cases per Capita"

a2_label = "State Population Density"



plt.xticks(rotation=90)

ax2=ax.twinx()

ax2.set_ylabel("State Population Density", fontsize=14)

# ax2.legend(loc="upper right")



ax.legend(loc="upper left")

ax.set_title("Cases per Capita and State Population Density", fontsize=20)

bar1 = ax.bar('USAState', 'CasesPerCapita', data=df, width=-.3, align='edge', label="Cases per Capita")

ax.set_ylabel("Cases per Capita", fontsize=14)

# ax.set_facecolor('xkcd:charcoal')

# ax.set_yscale(.005)

bar2 = ax2.bar('USAState', 'Density (sq mi)', data=df, color="peachpuff", width=.2, align='edge', label="State Population Density")





from matplotlib.patches import Rectangle



p1 = Rectangle((0, 0), 1, 1)

p2 = Rectangle((0, 0), 1, 1, fc="peachpuff")

ax.legend([p1, p2], [a1_label, a2_label], loc="upper right")



fig.tight_layout()



plt.savefig("CasesVsPopDensity.png")
us_geo = '../input/usa-states/usa-states.json'
us_map = folium.Map(location=[40.001626,-101.90605], zoom_start=4)



folium.Choropleth(

    geo_data=us_geo,

    data=df,

    columns=['USAState', 'Density (sq mi)'],

    key_on='feature.properties.name',

    fill_color='YlOrRd', 

    fill_opacity=0.7,

    line_opacity=0.2,

    legend_name="US Population Density",

    highlight=True

).add_to(us_map)



for i in range(0,len(df)):

    folium.Circle(([df.iloc[i]['Latitude'], df.iloc[i]['Longitude']]), 

#                         radius=df.iloc[i]['CasesPerCapita']*100000000,

#                         popup=df.iloc[i]['CasesPerCapita'], fill=True, 

                        radius=df.iloc[i]['TotalCases']*10,

                        popup=df.iloc[i]['TotalCases'], fill=True, 

                        fill_opacity=.3,

                        # fill_color=nearby_venues.iloc[i]['Marker Color'], 

                        # color=nearby_venues.iloc[i]['Marker Color']

                        ).add_to(us_map)





us_map