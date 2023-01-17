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
import pandas as pd

import warnings 

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)
house_data=pd.read_csv("../input/housedata/data.csv")

house_data.head()
house_data.info()
house_data["date"]=pd.to_datetime(house_data['date'], infer_datetime_format=True)

house_data.info()
house_data.corr(method ='pearson') 
neumaric=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_above','sqft_basement','yr_built','yr_renovated']



for i in neumaric:

    sns.lmplot(x = i, y ='price', data = house_data)

cities=house_data['city'].unique()

cities
house_data['city'].value_counts()
cities=['Seattle','Renton','Bellevue','Bellevue','Redmond','Kirkland','Issaquah','Kent','Auburn','Sammamish','Federal Way','Shoreline','Woodinville']

house_data_filtered=house_data[house_data.city.isin(cities)]

house_data_filtered.shape
sns.set(rc={'figure.figsize':(20,10)})

sns.boxplot(x="city", y="price", data=house_data_filtered)
q = house_data_filtered["price"].quantile(0.75)

house_data_final = house_data_filtered[(house_data_filtered["price"] < q)]

house_data_final.shape
sns.set_style('whitegrid') 

plot=sns.lmplot(x ='bedrooms', y ='price', data = house_data_final,col='city', hue ='city',height=5,col_wrap=5) 

sns.set_style('whitegrid') 

plot=sns.lmplot(x ='bathrooms', y ='price', data = house_data_final,col='city', hue ='city',height=5,col_wrap=5) 
sns.set_style('whitegrid') 

plot=sns.lmplot(x ='sqft_living', y ='price', data = house_data_final,col='city', hue ='city',height=5,col_wrap=5) 
sns.set_style('whitegrid') 

plot=sns.lmplot(x ='sqft_lot', y ='price', data = house_data_final,col='city', hue ='city',height=5,col_wrap=5)

sns.set_style('whitegrid') 

plot=sns.lmplot(x ='sqft_basement', y ='price', data = house_data_final,col='city', hue ='city',height=5,col_wrap=5)

sns.set_style('whitegrid') 

plot=sns.lmplot(x ='sqft_above', y ='price', data = house_data_final,col='city', hue ='city',height=5,col_wrap=5)

house_data_final.describe()
house_data_t=house_data_final[(house_data_final['bedrooms']<(3.235870+0.883879)) & (house_data_final['bedrooms']>(3.235870-0.883879))]

house_data_t=house_data_final[(house_data_final['bathrooms']<(1.959058+0.687286)) & (house_data_final['bathrooms']>(1.959058-0.687286))]

house_data_t=house_data_final[(house_data_final['sqft_living']<(1798.087319+673.396039)) & (house_data_final['sqft_living']>(1798.087319-673.396039))]

house_data_t=house_data_final[(house_data_final['sqft_above']<(1527.289855+619.584131)) & (house_data_final['sqft_above']>(1527.289855-619.584131))]

house_data_t.shape
features=['date','price','bedrooms','bathrooms','sqft_living','sqft_above','city']

house_data_t=house_data_t[features]

house_data_t.head()
p=['date','price']

seattle_data=house_data_t[house_data['city']=='Seattle']

seattle_data=seattle_data[p]

seattle_data.set_index('date',inplace=True)

seattle_data.head()

seattle_data.plot(grid=True)
