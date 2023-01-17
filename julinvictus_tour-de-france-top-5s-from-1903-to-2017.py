import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))



tour = pd.read_csv('../input/stages_TDF.csv')

tour.head()
tour.sort_values(by=['Winner']).head()

winners = tour["Winner"].value_counts().head(5)

winners.plot.bar()

print('Top 5 winning racers')
tour.sort_values(by=['Winner_Country']).head()

countries = tour["Winner_Country"].value_counts().head(5)

countries.plot.bar()

print('Top 5 winning countries')
longest = tour.loc[:,['Distance', 'Origin', 'Destination']]

longest.duplicated()

longest.drop_duplicates('Distance').sort_values(by=['Distance'], ascending=[False]).head(5)
tour.sort_values(by=['Origin','Destination']).head()

origin = tour["Origin"].value_counts().head(10)

destination = tour["Destination"].value_counts().head(10)

pop_cities = origin + destination 

pop_cities_bar = pop_cities.sort_values(ascending=[False]).head(5)

pop_cities_bar.plot.bar()

print('Top 5 most popular cities')