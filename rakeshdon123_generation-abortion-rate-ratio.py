# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Reading the CSV DataSet

data = pd.read_csv('/kaggle/input/abortion-2018/abortion_2018.csv')



#Overall Data

data



#Maximum in Generation Abortion Rate Ratio, with Year

print("Maximum in Generation Abortion Rate Ratio, with Year")

data.max()

#Minimum in Generation Abortion Rate Ratio, with Year

print("Minimum in Generation Abortion Rate Ratio, with Year")

data.min()

import matplotlib.pyplot as plt

%matplotlib inline





print("Overall inline Graph of the Data")

plt.plot(data.Period,data.General_abortion_rate)

plt.xlabel('Period')

plt.ylabel('Abortion Rate')

plt.show()



print("Five Largest Ratios in General_abortion Column ")

print("")



d1=data.nlargest(5, ['General_abortion_rate']) 

print(d1)



plt.plot(d1.Period, d1.General_abortion_rate)

plt.xlabel('Period')

plt.ylabel('General_abortion_rate')

plt.show()



d1=d1.describe()

print(d1)
print("Five Smallest Ratios in General_abortion Column ")

print("")

d2=data.nsmallest(5, ['General_abortion_rate']) 

d2=d2.sort_values(['General_abortion_rate'])

print(d2)



plt.plot(d2.Period, d2.General_abortion_rate)

plt.xlabel('Period')

plt.ylabel('General_abortion_rate')

plt.show()



d2=d2.describe()

print(d2)