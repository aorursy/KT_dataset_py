#Import the libraries we need: pandas, numpy, matplotlib and math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
#Read the database

data_country = pd.read_csv('/kaggle/input/TSD_February2015.csv')
data_country.head()
#We want to know how many countries are in the database.
con = len(data_country['REP'].value_counts())
con
#We want to know what countries are available in the database
val = pd.unique(data_country['REP']).tolist()
val
#We made the function that select a country from the column REP and we going to see the value of sales trought years.

def country_values(country):
    
    g = data_country.groupby(['REP','YEAR'])['VALUE'].agg(['sum','mean'])
    ndf = pd.DataFrame(g)
    f = ndf.reset_index('REP')
    l1 = f[f['REP'] == country]
    
    fig = plot.figure()
    ax = l1['sum'].plot()
    plot.show()
    
    country_values(input())
r = input('¿De qué país quiere analizar la información? ' )
country_values(r)