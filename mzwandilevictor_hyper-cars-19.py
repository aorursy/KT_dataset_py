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
# importing libraries 



import pandas as pd

import numpy as np

import matplotlib as mltt

import matplotlib.pyplot as plt

import seaborn as sns
# importing data



cars = pd.read_csv('../input/hyper-cars-2019/Hyper Cars 2019.csv')
cars.head(10)
cars.columns
cars['     Car Name'] ##  viewing cars names 
## Fixing all the misspelled car names



df = cars.replace('McLaren Sennaaaa', 'McLaren Senna')

df1 = df.replace('Koenigseeeggg Agera', 'Koenigsegg Agera RS')

df2 = df1.replace('Pagaaani Huayra BC',  'Pagani Huayra')

df3 = df2.replace('Bugatti Veyronre', 'Bugatti Veyron')
cars = df3
cars['     Car Name']
## I must replace hello with 0



cars['Displacement '] = cars['Displacement '].replace('Hello', np.nan)

cars['Displacement '] = cars['Displacement '].fillna(0)
## rounding of values 



cars['Displacement '] = cars['Displacement '].astype(float).round()
cars.head(10)
cars.columns
##  fixing all misspelled columns



df = cars.rename(columns = {'    Toppp-speeed': 'Top speed',

                           'Enginee': 'Engine'})
cars = df
cars.head(10)
## fixing all garbage values 



df = cars.replace('V8987654', 'V8')

df2 = df.replace('__', 'V12') # Was based on google search
cars = df2
cars.head(10)
# checking missing values



cars.isnull().any()
## the data have some missing values in the from columns like hp,Transmission and top speed
cars.shape
## The data have 10 rows and 7 columns

cars.columns
## gonna replace all Nan with mode number of each column



df = cars.fillna({'          hp': 1160.0,

                '   Transmission': 7.0,

                'Top speed': 350,

                })
df.head(10)
cars = df
cars['Displacement '] = cars['Displacement '].replace(0.0, np.nan)

cars['Displacement '] = cars['Displacement '].fillna(method = 'bfill')
## Lets view the data after all the cleaning and manipulation
cars.head(10)
cars.columns
# create a figure and axis

fig, ax = plt.subplots(figsize=(20,5))



# scatter plot

ax.scatter(cars['           Cost'], cars['Top speed'])

# set a title and labels

ax.set_title('cost and top speed relationship')

ax.set_xlabel('           Cost')

ax.set_ylabel('Top speed')
## checking the frequancies

cars.plot.hist(subplots=True, layout=(2,2), figsize=(10, 10), bins=20)
## This show the horsepower and top speed of cars 

cars[['          hp', 'Top speed']].plot.bar(figsize=(20,5))
## Data statistics 



cars.describe()
## showing all the columns relationship against car names 

plt.plot(cars['Displacement '], cars['     Car Name'], label=['Displacement '])
sns.pairplot(cars)
df = cars[['          hp', "Engine",'Top speed','Displacement ', '           Cost']]
df.head()
df.mode() ## this shows the data mode
df.corr()
sns.pairplot(df)
cars['           Cost'].value_counts().sort_index().plot.bar()
cars.columns