# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Firstly we load the data 



data = pd.read_csv("/kaggle/input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")
# To get basic informations of our data



data.info()
# Number of rows ans columns of our data



data.shape
# To see the column names



data.columns
# To make upper case the column names 



data.columns = map(str.lower, data.columns)



#data.columns = map(str.lower, data.columns)  

data['age'] = data['age'].astype('float')
# To see the data types of columns 



data.dtypes
# To see first 5 rows (default)



data.head() # it shows first 5 
# Too see rows as you want 



data.head(100)
# games column include year and season column so i delete it



# data.drop('games', axis=1, inplace=True)

data

# To see last 5 (default) rows



data.tail() 
data.tail(20)  # get the last rows as you want
# Get the columns increasing in a range 



data[100:2000:100]
# Get the Statistical Summary



data.describe()
# Standart Deviation of Columns



data.std()
# General Check for Missing Value



data.isnull().any().any()
# Checking the Missing Value for Columns



data.isnull().any()
# Counts of missing value of columns



data.isnull().sum()
# Fill the missing values with mean



data['height'].fillna((data['height'].mean()), inplace=True)

data['weight'].fillna((data['weight'].mean()), inplace=True)

data['age'].fillna((data['age'].mean()) , inplace = True )



# Medal feature type is object so fill with a string



data['medal'].fillna('no medal', inplace = True)



data5=data[data.medal=="no medal"]

data5
# Visualization the Outliers for age



data. boxplot(column='age')

plt.show()
data.age.describe()
# Calculation the Outliers Bound 



q1 = data['age'].quantile(0.25)

q3 = data.age.quantile(0.75)

iqr = q3 - q1



small =  q1 - 1.5*iqr

high = q3 + 1.5*iqr

small

high





outlier_values = data[np.logical_or(data['age'] < small ,data['age'] > high)]

outlier_values

# Checking the unique values in columns



print(data.age.unique())

print(data.year.unique())

# TIDY DATA 

# melt() method



melted = pd.melt(frame = data.head(100) , id_vars = 'name' , value_vars = ['team' , 'medal'] )

melted
data1 = data['team'] == 'Turkey'

data1

data1.value_counts(dropna = False)
data_turk = data [(data['team'] == 'Turkey') & (data['medal'] == 'Bronze') & (data['sex'] == 'M') & (data['year'] > 2000)]

data_turk