#import libraries 

#panda - to handle dataset

#matplotlib - to plot the results



import pandas as pd

import matplotlib.pyplot as plt
#Read dataset from the CSV file

gdp_growth_rate_data = pd.read_csv('../input/indian-gdp/IndiaGDPGrowthPercentageWorldBank.csv',header=None)





#drop rows with null values

gdp_growth_rate_data = gdp_growth_rate_data.dropna()
#Dataset after preprocessing

data = gdp_growth_rate_data.transpose()

data = data[4:15]
#Convert data column to numeric datatype

data[1] = pd.to_numeric(data[1])
#Plotting the figure using matplotlib

plt.figure (figsize = (20,10))



plt.plot(data[0],data[1],label='Year')



plt.xlabel('Year')

plt.ylabel('India - GDP Growth Rate %')



plt.legend()

plt.show()