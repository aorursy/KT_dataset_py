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
# basic imports
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
filepath = r'/kaggle/input/used-car-dataset-ford-and-mercedes/unclean focus.csv'
focus = pd.read_csv(filepath)
filepath = r'/kaggle/input/used-car-dataset-ford-and-mercedes/unclean cclass.csv'
cclass = pd.read_csv(filepath)
# removing duplicate car adverts
# also, there are duplicate columns because there were two possible locations where information could be for each car advert
# will combine in next step
focus = focus.drop_duplicates()
focus
# lots of missing data due to the way the website was layed out, had info in different places for each advert
# work around was to scrape all possible data, and combine it to fill in the gaps

focus['mileage'] = focus['mileage'].fillna(focus['mileage2'])
focus['fuel type'] = focus['fuel type2'].fillna(focus['fuel type'])
focus['engine size'] = focus['engine size2'].fillna(focus['engine size'])
focus.head()
focus = focus[['model','year','price','transmission','mileage', 'fuel type', 'engine size']]
focus = focus.rename(columns={"fuel type": "fuelType", "engine size": "engineSize"})
focus.head()
# looking for empty data
focus.isnull().sum()
# removing rows with empty data
focus = focus.dropna().reset_index(drop=True)

# remove 'Unknown' values in engine size and mileage categories
focus = focus[focus['engineSize'] != 'Unknown']
focus = focus[focus.mileage != 'Unknown']

# removing CC in strings from engine size e.g. 2000 CC engine, which is equivalent to 2L engine.
focus['engineSize'] = focus.iloc[:,6].str.replace('CC', '')
focus['engineSize'] = focus.iloc[:,6].str.replace(' cc', '')
focus['engineSize'] = focus.iloc[:,6].str.replace('T', '')
focus['engineSize'] = focus.iloc[:,6].str.replace(',', '.')

# removing commas from mileage
focus['mileage'] = focus.iloc[:,4].str.replace(',', '')
focus['price'] = focus.iloc[:,2].str.replace('£', '')
focus['price'] = focus.iloc[:,2].str.replace(',', '')

# converting price, mileage, engine size, and year into integer/float values
focus['price'] = focus.iloc[:,2].astype(int)
focus['mileage'] = focus.iloc[:,4].astype(int)
focus['engineSize'] = focus.iloc[:,6].astype(float)
focus.year = focus.year.astype(int)

# engine sizes are in varying units, so converting the CC values into L e.g. 5000 CC ---> 5L
focus['engineSize'] = focus['engineSize'].apply(lambda x: x/1000 if x > 20 else x)
focus['engineSize'] = focus['engineSize'].round(1)

focus
# checking values look as expected - and they do
focus.describe()
cclass = cclass.drop_duplicates()
cclass
cclass['mileage'] = cclass['mileage'].fillna(cclass['mileage2'])
cclass['fuel type'] = cclass['fuel type2'].fillna(cclass['fuel type'])
cclass['engine size'] = cclass['engine size2'].fillna(cclass['engine size'])
cclass
cclass = cclass[['model','year','price','transmission','mileage', 'fuel type', 'engine size']]
cclass = cclass.rename(columns={"fuel type": "fuelType", "engine size": "engineSize"})
cclass.head()
cclass.isnull().sum()
# removing rows with empty data
cclass = cclass.dropna().reset_index(drop=True)

# remove 'Unknown' values in engine size and mileage categories
cclass = cclass[cclass['engineSize'] != 'Unknown']
cclass = cclass[cclass.mileage != 'Unknown']

# removing CC in strings from engine size e.g. 2000 CC engine, which is equivalent to 2L engine.
cclass['engineSize'] = cclass.iloc[:,6].str.replace('CC', '')

# removing commas from mileage
cclass['mileage'] = cclass.iloc[:,4].str.replace(',', '')
cclass['price'] = cclass.iloc[:,2].str.replace('£', '')
cclass['price'] = cclass.iloc[:,2].str.replace(',', '')

# converting price, mileage, engine size, and year into integer/float values
cclass['price'] = cclass.iloc[:,2].astype(int)
cclass['mileage'] = cclass.iloc[:,4].astype(int)
cclass['engineSize'] = cclass.iloc[:,6].astype(float)
cclass.year = cclass.year.astype(int)

# converting the CC values into L e.g. 5000 CC ---> 5L
cclass['engineSize'] = cclass['engineSize'].apply(lambda x: x/1000 if x > 20 else x)
cclass['engineSize'] = cclass['engineSize'].round(1)

cclass
cclass.describe()
