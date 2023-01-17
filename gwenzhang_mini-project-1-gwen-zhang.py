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
#Create a DataFrame with the csv file.
df = pd.read_csv('../input/airbnb-new-york-city-listing-info/listings.csv', index_col=0, low_memory=False)
#select the necessary columns and drop the row contains NaN
select = df[['neighbourhood_cleansed','neighbourhood_group_cleansed','city','zipcode','accommodates','bedrooms','price','first_review','last_review']]
select = select.dropna(axis=0)
#quick peek
select
#Use first_review year as the year bucket the listed record belows to, then sum up the total number of bedrooms for each year.
bedrooms = select[['bedrooms','first_review','neighbourhood_group_cleansed']]
#Generate a new column 'Year' from column 'first_review'
bedrooms.insert(2,'year',pd.DatetimeIndex(bedrooms['first_review'], yearfirst=True).year)
#Generate result grouping by 'year'
totalrooms = bedrooms.groupby(['year']).sum()
#Exclude year 2020 because the data is partial.
totalrooms = totalrooms[:-1]

#The table of total number each year
totalrooms
#Plot the bedroom growing trend.
import matplotlib.pyplot as plt
import datetime as dt
#Plot the bedroom growing trend.
totalrooms.plot(y = 'bedrooms')
#YOY growth rate
YOY = totalrooms.pct_change()
YOY
neighborhood_totalrooms = bedrooms.groupby(['neighbourhood_group_cleansed', 'year']).sum()
#x = neighborhood_totalrooms.melt(id_vars=[''], value_vars=['bedrooms'])
neighborhood_totalrooms
rentperhead = select[['neighbourhood_group_cleansed','accommodates','price','first_review']]
rentperhead.insert(3,'year',pd.DatetimeIndex(rentperhead['first_review'], yearfirst=True).year)
#grouped_rentperhead = rentperhead.groupby(['neighbourhood_group_cleansed','year']).sum()
pricetotal = rentperhead[['neighbourhood_group_cleansed','year','price']]
#Transform the price from $xxx to xxx then insert it into data frame pricetotal
pricetotal.insert(3,'pricenumeric',pricetotal['price'].replace('[\$,]', '', regex=True).astype(float))
pricetotal = pricetotal[['neighbourhood_group_cleansed','year','pricenumeric']]
#groupby neighbourhood_group_cleansed and year
pricetotal = pricetotal.groupby(['neighbourhood_group_cleansed','year']).sum()
pricetotal
#Select all columns that related to average price per accomodates.
totalaccommodates = rentperhead[['neighbourhood_group_cleansed','year','accommodates']]
totalaccommodates = totalaccommodates.groupby(['neighbourhood_group_cleansed','year']).sum()
pricetotal.insert(1,'totalaccomdodates',totalaccommodates)
pricetotal
pricetotal['average price per accomodates'] = pricetotal['pricenumeric']/pricetotal['totalaccomdodates']
#pricetotal = pricetotal[['neighbourhood_group_cleansed']]
pricetotal
#Average price per accomodates for each district each year.
result = pricetotal[['average price per accomodates']]
#unstack then plot the result
result.unstack(level=0).plot(kind='bar', subplots=False).legend(loc='center left',bbox_to_anchor=(1.0, 0.8))