import numpy as np 

import pandas as pd 

US = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')

US=US.drop(['fips'], axis = 1) 
import matplotlib.pyplot as plt #plotting, math, stats

%matplotlib inline

import seaborn as sns 
US.tail()
x=US['date']

a=US['cases']

b=US['deaths']
fig = plt.figure(figsize=(18,7))

plt.plot(x, a)

plt.plot(x, b)



plt.title('Cases and Deaths from COVID-19') # Title

plt.xticks(US.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()