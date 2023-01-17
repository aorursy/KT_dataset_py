# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Kamonrut Vorrawongprasert
#6013633
df = pd.read_csv("../input/LemonadeFinal.csv")
#Find the average sales in unit
AvgSales = df['Sales'].mean()
AvgSales
#The table bellow show the records of the sale that are lower than the average
df[df.Sales < AvgSales]
df
#Plot the Scatter plot of the sales and temperature
%matplotlib inline
df.plot(kind = 'scatter', x = 'Sales', y = 'Temperature',color = 'black')
plt.xlabel('Sales')
plt.ylabel('Temperature')
plt.title('Sale and Temperature scatter plot')
plt.show()
#show the average of sales in unit in each day
avg_mon = df[df.Day == 'Monday']['Sales'].mean()
avg_tue = df[df.Day == 'Tuesday']['Sales'].mean()
avg_wed = df[df.Day == 'Wednesday']['Sales'].mean()
avg_thu = df[df.Day == 'Thursday']['Sales'].mean()
avg_fri = df[df.Day == 'Friday']['Sales'].mean()
avg_sat = df[df.Day == 'Saturday']['Sales'].mean()
avg_sun = df[df.Day == 'Sunday']['Sales'].mean()
df_class = pd.DataFrame([avg_sun,avg_mon,avg_tue,avg_wed,avg_thu,avg_fri,avg_sat])
df_class.index = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']

df_class.plot(kind='bar',stacked=False, figsize=(10,10), title="Average sales on each day")
print("The average sales on Monday are: ",avg_mon)
print("The average sales on Tuesday are: ",avg_tue)
print("The average sales on Wednesday are: ",avg_wed)
print("The average sales on Thursday are: ",avg_thu)
print("The average sales on Friday are: ",avg_fri)
print("The average sales on Saturday are: ",avg_sat)
print("The average sales on Sunday are: ",avg_sun)

