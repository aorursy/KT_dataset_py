# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file_path = '../input/weather-dataset/weatherHistory.csv'
dataset = pd.read_csv(file_path)

print(dataset.dtypes)
#changing into datetime object

dataset['Formatted Date'] = pd.to_datetime(dataset['Formatted Date'], utc=True)

print(dataset.dtypes)
dataset = dataset.set_index('Formatted Date')

dataset.head()
data_columns = ['Apparent Temperature (C)', 'Humidity']

df_monthly_mean = dataset[data_columns].resample('MS').mean()

df_monthly_mean.head()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

plt.figure(figsize=(14,6))

plt.title("Variation in Apparent Temperature and Humidity with time")

sns.lineplot(data=df_monthly_mean) 
df1 = df_monthly_mean[df_monthly_mean.index.month==1]

print(df1)

df1.dtypes
df2 = df_monthly_mean[df_monthly_mean.index.month==2]

df3 = df_monthly_mean[df_monthly_mean.index.month==3]

df4 = df_monthly_mean[df_monthly_mean.index.month==4]

df5 = df_monthly_mean[df_monthly_mean.index.month==5]

df6 = df_monthly_mean[df_monthly_mean.index.month==6]

df7 = df_monthly_mean[df_monthly_mean.index.month==7]

df8 = df_monthly_mean[df_monthly_mean.index.month==8]

df9 = df_monthly_mean[df_monthly_mean.index.month==9]

df10 = df_monthly_mean[df_monthly_mean.index.month==10]

df11 = df_monthly_mean[df_monthly_mean.index.month==11]

df12 = df_monthly_mean[df_monthly_mean.index.month==12]
# data to plot

n_groups = 11



# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.45

opacity = 0.9



rects1 = plt.bar(index, df1['Apparent Temperature (C)'], bar_width,

alpha=opacity,

color='r',

label='Apparent Temperature (C)')



rects2 = plt.bar(index + bar_width, df1['Humidity'], bar_width,

alpha=opacity,

color='g',

label='Humidity')



plt.rcParams["figure.figsize"] = (12,6)

plt.xlabel('Month Of january')

plt.ylabel('Temp & Humidity')

plt.title('Variation in Apparent Temperature and Humidity with Year')

plt.xticks(index + bar_width, ('2006', '2007', '2008', '2009','2010','2011','2012','2013','2014','2015','2016'))

plt.legend()

plt.show()
# data to plot

n_groups = 11



# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.45

opacity = 0.9



rects1 = plt.bar(index, df2['Apparent Temperature (C)'], bar_width,

alpha=opacity,

color='r',

label='Apparent Temperature (C)')



rects2 = plt.bar(index + bar_width, df2['Humidity'], bar_width,

alpha=opacity,

color='g',

label='Humidity')



plt.rcParams["figure.figsize"] = (12,6)

plt.xlabel('Month Of February')

plt.ylabel('Temp & Humidity')

plt.title('Variation in Apparent Temperature and Humidity with Year')

plt.xticks(index + bar_width, ('2006', '2007', '2008', '2009','2010','2011','2012','2013','2014','2015','2016'))

plt.legend()

plt.show()
# data to plot

n_groups = 11



# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.45

opacity = 0.9



rects1 = plt.bar(index, df3['Apparent Temperature (C)'], bar_width,

alpha=opacity,

color='r',

label='Apparent Temperature (C)')



rects2 = plt.bar(index + bar_width, df3['Humidity'], bar_width,

alpha=opacity,

color='g',

label='Humidity')



plt.rcParams["figure.figsize"] = (12,6)

plt.xlabel('Month Of March')

plt.ylabel('Temp & Humidity')

plt.title('Variation in Apparent Temperature and Humidity with Year')

plt.xticks(index + bar_width, ('2006', '2007', '2008', '2009','2010','2011','2012','2013','2014','2015','2016'))

plt.legend()

plt.show()
# data to plot

n_groups = 11



# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.45

opacity = 0.9



rects1 = plt.bar(index, df4['Apparent Temperature (C)'], bar_width,

alpha=opacity,

color='r',

label='Apparent Temperature (C)')



rects2 = plt.bar(index + bar_width, df4['Humidity'], bar_width,

alpha=opacity,

color='g',

label='Humidity')



plt.rcParams["figure.figsize"] = (12,6)

plt.xlabel('Month Of April')

plt.ylabel('Temp & Humidity')

plt.title('Variation in Apparent Temperature and Humidity with Year')

plt.xticks(index + bar_width, ('2006', '2007', '2008', '2009','2010','2011','2012','2013','2014','2015','2016'))

plt.legend()

plt.show()
# data to plot

n_groups = 11



# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.45

opacity = 0.9



rects1 = plt.bar(index, df5['Apparent Temperature (C)'], bar_width,

alpha=opacity,

color='r',

label='Apparent Temperature (C)')



rects2 = plt.bar(index + bar_width, df5['Humidity'], bar_width,

alpha=opacity,

color='g',

label='Humidity')



plt.rcParams["figure.figsize"] = (12,6)

plt.xlabel('Month Of May')

plt.ylabel('Temp & Humidity')

plt.title('Variation in Apparent Temperature and Humidity with Year')

plt.xticks(index + bar_width, ('2006', '2007', '2008', '2009','2010','2011','2012','2013','2014','2015','2016'))

plt.legend()

plt.show()
# data to plot

n_groups = 11



# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.45

opacity = 0.9



rects1 = plt.bar(index, df6['Apparent Temperature (C)'], bar_width,

alpha=opacity,

color='r',

label='Apparent Temperature (C)')



rects2 = plt.bar(index + bar_width, df6['Humidity'], bar_width,

alpha=opacity,

color='g',

label='Humidity')



plt.rcParams["figure.figsize"] = (12,6)

plt.xlabel('Month Of June')

plt.ylabel('Temp & Humidity')

plt.title('Variation in Apparent Temperature and Humidity with Year')

plt.xticks(index + bar_width, ('2006', '2007', '2008', '2009','2010','2011','2012','2013','2014','2015','2016'))

plt.legend()

plt.show()
# data to plot

n_groups = 11



# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.45

opacity = 0.9



rects1 = plt.bar(index, df7['Apparent Temperature (C)'], bar_width,

alpha=opacity,

color='r',

label='Apparent Temperature (C)')



rects2 = plt.bar(index + bar_width, df7['Humidity'], bar_width,

alpha=opacity,

color='g',

label='Humidity')



plt.rcParams["figure.figsize"] = (12,6)

plt.xlabel('Month Of July')

plt.ylabel('Temp & Humidity')

plt.title('Variation in Apparent Temperature and Humidity with Year')

plt.xticks(index + bar_width, ('2006', '2007', '2008', '2009','2010','2011','2012','2013','2014','2015','2016'))

plt.legend()

plt.show()
# data to plot

n_groups = 11



# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.45

opacity = 0.9



rects1 = plt.bar(index, df8['Apparent Temperature (C)'], bar_width,

alpha=opacity,

color='r',

label='Apparent Temperature (C)')



rects2 = plt.bar(index + bar_width, df8['Humidity'], bar_width,

alpha=opacity,

color='g',

label='Humidity')



plt.rcParams["figure.figsize"] = (12,6)

plt.xlabel('Month Of August')

plt.ylabel('Temp & Humidity')

plt.title('Variation in Apparent Temperature and Humidity with Year')

plt.xticks(index + bar_width, ('2006', '2007', '2008', '2009','2010','2011','2012','2013','2014','2015','2016'))

plt.legend()

plt.show()
# data to plot

n_groups = 11



# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.45

opacity = 0.9



rects1 = plt.bar(index, df9['Apparent Temperature (C)'], bar_width,

alpha=opacity,

color='r',

label='Apparent Temperature (C)')



rects2 = plt.bar(index + bar_width, df9['Humidity'], bar_width,

alpha=opacity,

color='g',

label='Humidity')



plt.rcParams["figure.figsize"] = (12,6)

plt.xlabel('Month Of September')

plt.ylabel('Temp & Humidity')

plt.title('Variation in Apparent Temperature and Humidity with Year')

plt.xticks(index + bar_width, ('2006', '2007', '2008', '2009','2010','2011','2012','2013','2014','2015','2016'))

plt.legend()

plt.show()
# data to plot

n_groups = 11



# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.45

opacity = 0.9



rects1 = plt.bar(index, df10['Apparent Temperature (C)'], bar_width,

alpha=opacity,

color='r',

label='Apparent Temperature (C)')



rects2 = plt.bar(index + bar_width, df10['Humidity'], bar_width,

alpha=opacity,

color='g',

label='Humidity')



plt.rcParams["figure.figsize"] = (12,6)

plt.xlabel('Month Of October')

plt.ylabel('Temp & Humidity')

plt.title('Variation in Apparent Temperature and Humidity with Year')

plt.xticks(index + bar_width, ('2006', '2007', '2008', '2009','2010','2011','2012','2013','2014','2015','2016'))

plt.legend()

plt.show()
# data to plot

n_groups = 11



# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.45

opacity = 0.9



rects1 = plt.bar(index, df11['Apparent Temperature (C)'], bar_width,

alpha=opacity,

color='r',

label='Apparent Temperature (C)')



rects2 = plt.bar(index + bar_width, df11['Humidity'], bar_width,

alpha=opacity,

color='g',

label='Humidity')



plt.rcParams["figure.figsize"] = (12,6)

plt.xlabel('Month Of November')

plt.ylabel('Temp & Humidity')

plt.title('Variation in Apparent Temperature and Humidity with Year')

plt.xticks(index + bar_width, ('2006', '2007', '2008', '2009','2010','2011','2012','2013','2014','2015','2016'))

plt.legend()

plt.show()
warnings.filterwarnings("ignore")

plt.figure(figsize=(14,6))

plt.title("Variation in Apparent Temperature and Humidity with time")

sns.lineplot(data=df12)
from scipy.stats import ttest_1samp

import numpy as np

ages_mean = np.mean(df_monthly_mean['Apparent Temperature (C)'])

print(ages_mean)

tset, pval = ttest_1samp(df_monthly_mean['Apparent Temperature (C)'], 11)

print('p-values',pval)

if pval < 0.05:    # alpha value is 0.05 or 5%

   print(" we are rejecting null hypothesis")

else:

  print("we are accepting null hypothesis")