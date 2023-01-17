#lib



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



df  = pd.read_csv('../input/szeged-weather/weatherHistory.csv')

df.head()
#data Preprocessing

#check for null/ NaN values

df.isnull().any().any()

#check for missing values

df.isnull().values.any()
#Visualise 

plt.title('Precip Type Vs. Temperature')

plt.xlabel('Temperature')

plt.ylabel('Precip Type')

plt.plot(

        df['Temperature (C)'].values, df['Precip Type'].values,'or',alpha=0.5

)

plt.show()


