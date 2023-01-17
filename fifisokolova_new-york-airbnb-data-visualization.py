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







import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns  

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics



path = "/kaggle/input/new-york-city-airbnb-open-data/"



filename_read = os.path.join(path, "AB_NYC_2019.csv")

df=pd.read_csv(filename_read, na_values=['NA','?','NaN'])

df = df.reindex(np.random.permutation(df.index))

df=df.interpolate(method='linear')

print(df.head(3))

print( list(df.columns))# show the features and label 

print( df.shape)

df.dtypes



df.isnull().sum()

df.fillna('0',inplace=True)



df.isnull().sum()
#Plots to understand the data better



#List of the columns containing numerical values 

numerical = [

   'latitude', 'longitude', 'price', 'minimum_nights', 'calculated_host_listings_count','availability_365', 'number_of_reviews', 'reviews_per_month'

]

#List of the columns containing categorical values 

categorical = [

  'neighbourhood_group', 'room_type']

#dropping columns that are not needed for the model

df.drop(columns=['name', 'host_id','host_name', 'neighbourhood','id','last_review',])

df = df[numerical + categorical]

df.shape



#Heatmap for the ralation between all the columns

relation = df.corr(method='kendall')

sns.heatmap(relation, annot=True)

df.columns


#Countplot for the count of airBnBs with the same price

sns.set(style='whitegrid', palette="deep", font_scale=1, rc={"figure.figsize": [8, 5]})

sns.distplot(

    df['price'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}

).set(xlabel='Price', ylabel='Count');

df[numerical].hist(bins=15, figsize=(15, 6), layout=(2, 4));

#plot for room_type

sns.countplot(df['room_type']);

#plor for neighbourhood_group

sns.countplot(df['neighbourhood_group']);





#plot for the relation betweeb room_type and price

sns.scatterplot(x=df['room_type'], y=df['price']);

#plot for the relation betweeb neighbourhood_group and price

sns.scatterplot(x=df['neighbourhood_group'], y=df['price']);

#plot for the relation betweeb minimum_nights and price

sns.scatterplot(x=df['minimum_nights'], y=df['price']);



#plot for the relation betweeb availability_365 and price

sns.scatterplot(x=df['availability_365'], y=df['price']);



#More interesting plot for the relation between availability_365 and price showing both their individual plots and the relation

sns.jointplot(x=df['availability_365'], y=df['price']);



#Map for longtitude, latitude and neighbourhood_group

sns.scatterplot(df.longitude,df.latitude,hue=df.neighbourhood_group)

plt.ioff()


#Map for longtitude, latitude and room_type

sns.scatterplot(df.longitude,df.latitude,hue=df.room_type)

plt.ioff()