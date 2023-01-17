import pandas as pd

cities = pd.read_excel('../input/cities/zip_codes.xlsx')
#this is how our dateframe looks like

cities.sample(10)
#to save distinct values of the column 'VILLE' for futher use

import numpy as np

#save it as a dataframe

distinct_cities = pd.DataFrame(np.unique(cities.VILLE))

distinct_cities

#we got 33 cities
#then we can save it as we want, for example here I save it as a csv file without indices

distinct_cities.to_csv('distinct_cities.csv', index = False)