import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # statistical data visualization
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
airpi = pd.read_csv("../input/AirPi Data - AirPi.csv")
airpi.head()
airpi.hist(color='red', figsize=(15, 9), bins=50)
airpi.drop(airpi[airpi['Temperature-DHT [Celsius]'] < 0].index, inplace=True)
airpi.drop(airpi[airpi['Relative_Humidity [%]'] < 0].index, inplace=True)
airpi.drop(airpi[airpi['Relative_Humidity [%]'] > 100].index, inplace=True)
airpi.hist(color='green', figsize=(15, 9), bins=50)
from pandas.plotting import scatter_matrix
scatter_matrix(airpi, alpha=0.4, figsize=(18, 18), diagonal='kde')
