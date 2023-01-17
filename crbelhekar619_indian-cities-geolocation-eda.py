import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/geolocations-of-indian-cities/geolocations-indian-cities.csv', encoding = "ISO-8859-1")

df.head()
df.info()
len(df['ASCII Name'].unique())