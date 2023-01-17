import pandas as pd



df = pd.read_csv(r'../input/weather_place.data.csv')

df.head()
df.drop(columns='Unnamed: 0', inplace=True)

df.head()
df.columns = ['Temperature','Date','Parameters','Place']

df.head()
df.isnull().values.any()
tf = df[:10]

tf.head()
import numpy as np

tf.replace(0.00 , value=np.NaN)
tf.replace(0.00 , tf.mean())