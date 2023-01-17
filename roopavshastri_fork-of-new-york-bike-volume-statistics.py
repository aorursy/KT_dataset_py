#Importing the Libraries

import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats



#Importing the CSV File

df = pd.read_csv('C:/Users/Vinod/Desktop/Chicago CRimes/nyc-east-river-bicycle-counts.csv')



#Creating a new column the average of High and Low Temp of the Day to find the realtionship between

#the Volumne of the Bikes and the Tempertaure

df['avgtemp']=df[['Low Temp (°F)','High Temp (°F)']].mean(axis=1)









#Eliminating null and outliers in the Percipitation column

data_columns=['Precipitation']

df = (df.drop(data_columns, axis=1)

          .join(df[data_columns].apply(pd.to_numeric, errors='coerce')))

df =df[df[data_columns].notnull().all(axis=1)]



plt.scatter(df['Precipitation'],df['Total'])



plt.xlabel('Precipitation')

plt.ylabel('Total Number Of Bicyclists')

plt.show()



#Calculate the sope,intercept and r-squared value

from scipy import stats

slope,intercept,r_value,p_value,std_err=stats.linregress(df['avgtemp'],df['Total'])

print(r_value**2)



def predict(x):

    return slope *x+intercept

fitline=predict(df['avgtemp'])



#Scatter Plot depicting the relationship between Averge Temp Of the Day and Bike Volume



plt.scatter(df['avgtemp'],df['Total'])

plt.plot(df['avgtemp'],fitline,c='r')

plt.xlabel('Avg Temp of the day')

plt.ylabel('Total Number Of Bicyclists')

plt.show()



#2) What is the top bridge in terms of bike load?

df[['Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge']].sum().sort_values().plot(kind='barh')

plt.show()



#3) Average Bike volumes for the day

#The red line represents the median of the data, and the box represents the bounds of the

#1st and 3rd quartiles. Here the lease bike volume is around 4800 and maximum value being 24000.Median is  15000, only 25% of the times the bike volumes are greater than the median.



import statsmodels.api as sm

plt.boxplot(df['Total'])

plt.show()




