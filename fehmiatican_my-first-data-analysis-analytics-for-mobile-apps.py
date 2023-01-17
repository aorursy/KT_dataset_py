# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Starting with importing our .csv file
data = pd.read_csv('../input/AppleStore.csv')
data.info() #tried to understand what is inside the csv file
data.head() #with info and head codes, I can understand what is inside
data.columns # as we see here which columns we have
data.corr() #tried to understand any correlations between them 
import matplotlib.pyplot as plt
import seaborn as sns
f,ax = plt.subplots(figsize= (15,15))
sns.heatmap(data.corr(),annot = True,square=True)
plt.show()
data[data.price!=0]# here we are filtering the priced apps

data_priced = data[data.price!=0]
data_priced.head()
data_priced[data_priced.user_rating>4] # we applied second filter by user rating
data_priced_liked = data[(data.price != 0) & (data.user_rating>4)] #filtering with and(&) logic 
data_priced_liked # It works! 
data_priced_liked.prime_genre.unique() 
dpl = data_priced_liked
dpl_utilities = dpl[dpl.prime_genre == 'Utilities']
dpl_Games = dpl[dpl.prime_genre == 'Games']
dpl_Reference = dpl[dpl.prime_genre == 'Reference']
dpl_Business = dpl[dpl.prime_genre == 'Business']
dpl_Health_Fitness = dpl[dpl.prime_genre == 'Health & Fitness']
dpl_Weather = dpl[dpl.prime_genre == 'Weather']
dpl_Photo_Videos = dpl[dpl.prime_genre == 'Photo & Video']
dpl_Education = dpl[dpl.prime_genre == 'Education']
dpl_Music = dpl[dpl.prime_genre == 'Music']
dpl_Medical = dpl[dpl.prime_genre == 'Medical']
dpl_Navigation = dpl[dpl.prime_genre == 'Navigation']
dpl_Lifestyle = dpl[dpl.prime_genre == 'Lifestyle']
dpl_Shopping = dpl[dpl.prime_genre == 'Shopping']
dpl_Productivity = dpl[dpl.prime_genre == 'Productivity']
dpl_Finance = dpl[dpl.prime_genre == 'Finance']
dpl_Entertainment = dpl[dpl.prime_genre == 'Entertainment']
dpl_Book = dpl[dpl.prime_genre == 'Book']
dpl_Food_Drink = dpl[dpl.prime_genre == 'Food & Drink']
dpl_Social_Networking = dpl[dpl.prime_genre == 'Social Networking']
dpl_Sports = dpl[dpl.prime_genre == 'Sports']
dpl_News = dpl[dpl.prime_genre == 'News']
dpl_Travel = dpl[dpl.prime_genre == 'Travel']
dpl_Catalogs = dpl[dpl.prime_genre == 'Catalogs']

dpl_Games.price.plot(kind='line',color ='g',label = 'Games',linewidth=1,alpha = 0.5,grid = True,linestyle = '-')
dpl_Education.price.plot(kind='line',color='r',label = 'Education')
dpl_Sports.price.plot(kind='line',color='b',label = 'Sports')
plt.legend()    
plt.xlabel('x axis')              
plt.ylabel('Price_USD')
plt.title('Line Plot')           
plt.show()
dpl_Games.plot(kind='scatter',x='user_rating_ver', y= 'rating_count_tot',color ='g')
plt.title("rating count vs rating for games")
dpl_Education.plot(kind='scatter',x='user_rating_ver', y= 'rating_count_tot',color ='r')
plt.title("rating count vs rating for Education")
dpl_Sports.plot(kind='scatter',x='user_rating_ver', y= 'rating_count_tot',color ='b')
plt.title("rating count vs rating for Sports")

dpl_Games.price.plot(kind = 'hist',bins = 10,figsize = (20,20),color ='r')
dpl_Education.price.plot(kind = 'hist',bins = 10,figsize = (20,20),color ='g')
dpl_Sports.price.plot(kind = 'hist',bins = 10,figsize = (20,20),color ='b')
plt.show()
for i,price in dpl_Games[['price']].iterrows():
    print(i," : ",price,dpl_Games.track_name[i])
