import pandas as pd

import numpy as np

df = pd.read_csv(r'../input/weather_place.data.csv' )

df.drop(columns = 'Unnamed: 0', inplace = True)

df.head()
df.tail()
df.columns = ['Temp','Date','Parameters','Place']

df.head(3)
tf = df[:10]

tf.head(10)
tf.pivot(index = 'Place', columns = 'Date')
tf.pivot_table(index='Place',columns ='Date',margins = True,aggfunc = np.sum)