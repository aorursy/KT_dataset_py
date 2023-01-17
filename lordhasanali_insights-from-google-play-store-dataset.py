# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../inp ut/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.preprocessing import MinMaxScaler

import os

print(os.listdir("../input"))

scale  = MinMaxScaler()

# Any results you write to the current directory are saved as output.
googleps_data = pd.read_csv('../input/googleplaystore.csv')

user_reviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')
(googleps_data.isna().sum()/len(googleps_data))*100
#Imputing the nan by shifting to the right adjacent column value of Life Made Wi-Fi Touchscreen Photo Frame



googleps_data.loc[10472,'Android Ver'] = googleps_data.loc[10472,'Current Ver']

googleps_data.loc[10472,'Current Ver'] = googleps_data.loc[10472,'Last Updated']

googleps_data.loc[10472,'Last Updated'] = googleps_data.loc[10472,'Genres']

googleps_data.loc[10472,'Genres'] = googleps_data.loc[10472,'Content Rating']

googleps_data.loc[10472,'Content Rating'] = googleps_data.loc[10472,'Price']

googleps_data.loc[10472,'Price'] = googleps_data.loc[10472,'Type']

googleps_data.loc[10472,'Type'] = googleps_data.loc[10472,'Installs']

googleps_data.loc[10472,'Installs'] = googleps_data.loc[10472,'Size']

googleps_data.loc[10472,'Size'] = googleps_data.loc[10472,'Reviews']

googleps_data.loc[10472,'Reviews'] = googleps_data.loc[10472,'Rating']

googleps_data.loc[10472,'Rating'] = googleps_data.loc[10472,'Category']

#Photo-Frams lie in the PHOTOGAPHY CATEGORY

googleps_data.loc[10472,'Category'] = 'PHOTOGRAPHY'

#Doing little feature engineering to check correlation between 
googleps_data['Rating']= googleps_data["Rating"].astype(np.float64)
#Imputing for missing Android Version with "4.0 and above because of the apps with that rating and installs have that Android Version"

googleps_data.loc[4453,'Android Ver'] = '4.0 and up'

googleps_data.loc[4490,'Android Ver'] = '4.0 and up'
####### Since most of the data is in version 1.0, we'll impute it by version 1.0

for item in list(googleps_data[googleps_data['Current Ver'].isna()].index):

    googleps_data.loc[item,'Current Ver'] = '1.0'

    

googleps_data.loc[10472,'Genres'] ='Photography'

#The Price is zero therefore the Type is Free

googleps_data.loc[9148,'Type'] = 'Free'

#converting the column to int64

googleps_data['Reviews'] = googleps_data['Reviews'].astype('int64')
googleps_data['Installs'] = googleps_data['Installs'].apply(lambda x : x.replace('+',''))

googleps_data['Installs'] = googleps_data['Installs'].apply(lambda x : x.replace (',',''))

googleps_data[googleps_data['App'].str.contains('Theme')]['Android Ver'].value_counts()
googleps_data.loc[9148,'Type'] = 'Free'

googleps_data.loc[9148,'Rating'] = 0

for it in list(googleps_data[googleps_data.Rating.isna()].index):

    googleps_data.loc[it,'Rating'] = 0

    
#copying the dataset

copy_dataset = googleps_data.copy()
#Removing all the duplicate values

copy_dataset.drop_duplicates(subset = 'App', keep='first', inplace=True)
#Best CG Photography is a photography app and can be used by anyone 

copy_dataset.loc[7312,'Content Rating'] = 'Everyone'

#DC UNiverse Online Maps is a topic most popular among teens

copy_dataset.loc[8266,'Content Rating'] = 'Teen'
#converting to required data types

copy_dataset['Rating'] = copy_dataset['Rating'].astype('float64')

copy_dataset['Price'] = copy_dataset['Price'].apply(lambda x : x.replace('$',''))

copy_dataset['Price'] = copy_dataset['Price'].astype('float64')

copy_dataset['Installs'] = copy_dataset['Installs'].astype('int64')



#Replacing all the sizes having less than 1MB with 1MB

copy_dataset.loc[copy_dataset['Size'].str.contains('k'), 'Size']  = '1M'

#Removing the M and k so that the number becomes comparable

copy_dataset['Size'] =  copy_dataset['Size'].apply(lambda x : x.replace('M',''))

copy_dataset['Size'] =  copy_dataset['Size'].apply(lambda x : x.replace('k',''))



#Let's assume apps whose size varies with device are of 0 just to do some EDA

copy_dataset.loc[copy_dataset['Size'].str.contains('Varies'), 'Size']  = 0

copy_dataset['Size'] = copy_dataset['Size'].astype('float64')

## Reducing the Genres into one simple category

copy_dataset['Genres'] = copy_dataset['Genres'].apply(lambda x : x.split(";")[0])

copy_dataset['Genres'].value_counts()
def count_relationship(a,b):

    df = copy_dataset.groupby(a)[b].count()

    df = df.reset_index()

    df = df.sort_values(by=[b])

    return df.plot.barh(x=a,y=b, figsize = (12,10))
def sum_relationship(a,b):

    df = copy_dataset.groupby(a)[b].sum()

    df = df.reset_index()

    df[b] = (df[b]*100)/sum(copy_dataset[b])

    df = df.sort_values(by=[b])

    return df.plot.barh(x=a,y=b, figsize = (12,20))
# Apps with most installs genre, size, cateory ,reviews and ratingss

count_relationship('Category','Installs')
for a in (copy_dataset['Category'].unique()):

    if (len(copy_dataset[copy_dataset['Category'] == a].groupby('Genres').count()) > 1) :

        print (a)
fig, axarr = plt.subplots(4, 2, figsize=(26, 15))

plt.subplots_adjust(top=1.2, hspace=0.5)



for i,col in enumerate(['GAME','FAMILY']):

    axarr[0][i].set_title(col)

    axarr[0][i].set(ylabel = 'Number of Apps' )

    axarr[1][i].set(ylabel = 'Total Installs')

    axarr[2][i].set(ylabel = 'Average Ratings')

    axarr[3][i].set(ylabel = 'Number of people who rated')

    copy_dataset[copy_dataset['Category'] == col].groupby('Genres')['Installs'].count().plot.bar(ax = axarr[0][i])

    copy_dataset[copy_dataset['Category'] == col].groupby('Genres')['Installs'].sum().plot.bar(ax = axarr[1][i])    

    copy_dataset[copy_dataset['Category'] == col].groupby('Genres')['Rating'].mean().plot.bar(ax = axarr[2][i])

    copy_dataset[copy_dataset['Category'] == col].groupby('Genres')['Rating'].count().plot.bar(ax = axarr[3][i])    

plt.figure(figsize=(12,6))

pd.DataFrame(copy_dataset[copy_dataset['Type'] == 'Paid'].groupby(['Category','Genres']).mean()['Rating'] ).sort_values(['Rating'],ascending = False).plot.bar(figsize=(24,6))   

plt.title('Ratings of Paid Apps Grouped by Category and Genres')
corr = copy_dataset.corr()

corr.style.background_gradient(cmap='coolwarm')