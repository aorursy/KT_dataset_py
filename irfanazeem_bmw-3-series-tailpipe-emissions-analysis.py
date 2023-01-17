import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import pandas as pd

import numpy as np

import re

import string

from sklearn import ensemble

from sklearn import metrics

import seaborn



import numpy as np

import matplotlib.pyplot as plt



from pandas.tools import plotting

from pandas import Timestamp
bmw = pd.ExcelFile('../input/Tailpipe emissions BMW 3series  sedan.xls')

bmw.sheet_names

[bmw,"Vets23654"]

df = bmw.parse("Vets23654")

df.head(5)
df.describe()
#Descriptive statistics using Pandas Describe

grouped_data = df.groupby(['Time (Sec)','THC_tailpipe (g/s)','CO_tailpipe (g/s)','NOx_tailpipe (g/s)'])

grouped_data['Vehicle_speed (km/h)'].describe().unstack()
#grouped_data['Time (Sec)'].mean().reset_index()

grouped_data.aggregate(np.mean).reset_index()
#There are at least two ways of doing this using our grouped data. First, Pandas have the method mean;

grouped_data = df.groupby(['Time (Sec)','THC_tailpipe (g/s)','CO_tailpipe (g/s)','NOx_tailpipe (g/s)'])

grouped_data['Vehicle_speed (km/h)'].describe().unstack()



grouped_data['Vehicle_speed (km/h)'].mean().reset_index()

#grouped_data['Vehicle_speed (km/h)'].aggregate(np.mean).reset_index()
grouped_data['Vehicle_speed (km/h)'].aggregate(np.mean).reset_index()
#Select a random subset of 898 without replacement

sample1 = df.take(np.random.permutation(len(df))[:898])

sample1.head()
df.sample(n=10)
#And a random 50% of the DataFrame with replacement:

sample2 = df.sample(frac=0.50, replace=True)

sample2.head(5)
#And a random 50% of the DataFrame with No replacement:

sample3 = df.sample(frac=0.50, replace=False)



#specify replace=False if you want sampling without replacement. 

#Otherwise this method can potentially sample the same row multiple times

sample3.head(5)
#check in data has any missing values

sample3.isnull().values.any()
#check in data has any missing values

sample3.isnull().sum().sum()
sample3.isnull().head(5)
#sample3.shape

sample4 = sample3[lambda df: df.columns[0:5]]

sample4.head(2)
time_speed = sample4.plot.scatter(x='Time (Sec)', y='Vehicle_speed (km/h)');
time_thc = sample4.plot.scatter(x='Time (Sec)', y='THC_tailpipe (g/s)');
time_co = sample4.plot.scatter(x='Time (Sec)', y='CO_tailpipe (g/s)');
speed = sample4['Vehicle_speed (km/h)'].plot(x_compat=True) 
from matplotlib import cm

plt.figure()

sample4.plot(colormap=cm.cubehelix)
no_tailpipe = sample4['NOx_tailpipe (g/s)'].plot(x_compat=True)
speed1 = sample4['Vehicle_speed (km/h)'].plot(legend=False) 
scat = sample4.plot.scatter(x='THC_tailpipe (g/s)', y='CO_tailpipe (g/s)');
time_thc = sample4.plot.scatter(x='Time (Sec)', y='THC_tailpipe (g/s)');
time_co = sample4.plot.scatter(x='Time (Sec)', y='CO_tailpipe (g/s)');
time5 = sample4['Time (Sec)'].plot.area(stacked=False);
time_speed_co = sample4.plot.scatter(x='Time (Sec)', y='Vehicle_speed (km/h)', c = 'CO_tailpipe (g/s)');
#rename the column names coz origina names were giving error when run the codes

sample4.columns = ['time','speed','THC','CO',"NOx"]
sample4[(sample4.speed  >= 39.9) & (sample4.THC  >= 0.0005) &

        (sample4.CO  >= 0.0200) & (sample4.NOx >= 0.0004)]
plotting.scatter_matrix(sample4[['time', 'speed','THC', 'CO', 'NOx']]) 
seaborn.pairplot(sample4, vars=['time', 'speed','THC', 'CO', 'NOx'],kind='reg')
from statsmodels.formula.api import ols



result = ols(formula='time ~ THC + CO + CO * speed',

                data=sample4).fit()    

print(result.summary())  
low = sample4[sample4['speed'] <= 39.9]

high= sample4[sample4['speed'] > 39.9]



low.mean()
high.mean()
sample4.mean()
from scipy import stats

stats.ttest_ind(low['THC'], high['THC'])
stats.ttest_ind(low['speed'], high['speed'])
stats.ttest_ind(low['CO'], high['CO'])
stats.ttest_ind(low['time'], high['time'])
stats.ttest_ind(low['NOx'], high['NOx'])