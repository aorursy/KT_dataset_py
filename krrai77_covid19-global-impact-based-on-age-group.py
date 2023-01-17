# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from matplotlib import pyplot as plt

import seaborn as sns

import plotly.express as px


covidcountry = pd.read_csv("/kaggle/input/countryinfo/covid19_merged.csv")

covidcountry.columns
covidcouage=covidcountry[['country','age_0_to_14_years_percent',

       'age_15_to_64_years_percent', 'age_over_65_years_percent','latitude',

       'longitude']].copy()

covidcouage=covidcouage.dropna()

covidcouage.age_0_to_14_years_percent=covidcouage.age_0_to_14_years_percent.astype(int)

covidcouage.age_15_to_64_years_percent=covidcouage.age_15_to_64_years_percent.astype(int)

covidcouage.age_over_65_years_percent=covidcouage.age_over_65_years_percent.astype(int)

covidcouage


fig = px.bar(covidcouage[['country','age_0_to_14_years_percent']].sort_values('age_0_to_14_years_percent', ascending=False), 

                        y = "age_0_to_14_years_percent", x= "country", color='age_0_to_14_years_percent', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="Global Effect of Covid19 on Children")



fig.show()
fig = px.bar(covidcouage[['country','age_15_to_64_years_percent']].sort_values('age_15_to_64_years_percent', ascending=False), 

                        y = "age_15_to_64_years_percent", x= "country", color='age_15_to_64_years_percent', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="Global Effect of Covid19 on People between 15 to 64 years")



fig.show()
fig = px.bar(covidcouage[['country','age_over_65_years_percent']].sort_values('age_over_65_years_percent', ascending=False), 

                        y = "age_over_65_years_percent", x= "country", color='age_over_65_years_percent', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="Global Effect of Covid19 on Senior Citizens")



fig.show()
covidage=pd.read_csv("/kaggle/input/covidindia/AgeGroupDetails.csv")

covidage=covidage.drop([9], axis=0)

covidage
plt.figure(figsize=(23,10))

plt.bar(covidage.AgeGroup, covidage.TotalCases,label="Age Group")

plt.xlabel('Age Group')

plt.ylabel("Cases")

plt.legend(frameon=True, fontsize=25)

plt.title('Affected Age Group in India',fontsize=30)

plt.show()
plt.figure(figsize=(23,10))

plt.pie(covidage['TotalCases'], labels=covidage['TotalCases'])

#plt.show()



plt.legend(covidage['AgeGroup'],

          title="Age Group",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1))

plt.title("Affected Age Group Distribution in India", size = 30)





my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

 

plt.show()
covidage1 = covidcouage.drop(["latitude","longitude"],axis=1)

sns.pairplot(covidage1)
import folium 

from folium import plugins

print("Global Covid19 spread among population in the age group of 15 to 64 years")

map = folium.Map(location=[20, 80], zoom_start=2,tiles='Stamen Toner')



for lat, lon, value, name in zip(covidcouage['latitude'], covidcouage['longitude'], covidcouage['age_15_to_64_years_percent'], covidcouage['country']):

    folium.CircleMarker([lat, lon],

                        radius=value*0.1,

                        popup = ('<strong>country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>age_15_to_64_years_percent</strong>: ' + str(value) + '<br>'), color='red',                       

                        fill_color='blue',                  

                        fill_opacity=0.3 ).add_to(map)

map
