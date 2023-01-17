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
Restaurants=pd.read_csv('/kaggle/input/zomato-restaurants-hyderabad/Restaurant names and Metadata.csv')
Reviews=pd.read_csv('/kaggle/input/zomato-restaurants-hyderabad/Restaurant reviews.csv')
Restaurants.head(5)
Reviews.head(5)
#find the most popular 5 collections
import itertools
import collections
from collections import Counter
import operator
restaurant_collections=Restaurants['Collections'].dropna()
list_of_collections=list(itertools.chain.from_iterable(restaurant_collections.str.split(',').to_list()))
list_of_collections=list(map(str.strip,list_of_collections))
top_5_collections=[a[0] for a in Counter(list_of_collections).most_common(5)]
#count_dict=dict(Counter(list_of_collections))
#quick check to see if there any duplicate keys
#order=collections.OrderedDict(sorted(count_dict.items(),key=lambda x:x[1]))


print(top_5_collections)
#find the most popular and least popular 5 cuisines
import itertools
import collections
from collections import Counter
import operator
restaurant_cuisines=Restaurants['Cuisines'].dropna()
list_of_cuisines=list(itertools.chain.from_iterable(restaurant_cuisines.str.split(',').to_list()))
list_of_cuisines=list(map(str.strip,list_of_cuisines))
#print(list_of_cuisines)
popular_cuisines=[a[0] for a in Counter(list_of_cuisines).most_common()][:5]
print(popular_cuisines)
count_dict_cuisines=dict(Counter(list_of_cuisines))
order_by_pop_cusines=collections.OrderedDict(sorted(count_dict_cuisines.items(),key=lambda x:x[1]))

least_popular_cuisines=[a for a in order_by_pop_cusines][:10]
print(least_popular_cuisines)
Restaurants.head(5)
#lets create new columns based on collections aand cuisines
#1)create a new flag column for most popular and least popular cuisine
[ exec("Restaurants['"+a.replace(' ','_')+"']"+"=0") for a in popular_cuisines]
#restaurants that has atleast one of the least popular cusines will be marked
Restaurants['least_popular_cuisine']=0
#restaurants that are featured in atleast one of the popular collections will be featured here
Restaurants['featured_in_pop_collections']=0


#dynamically populate the indicator columns
#Restaurants.apply(lambda a:[exec("Restaurants.loc["+ str(a.name)+","+"'"+b.replace(" ","_")+"']=1") for b in a['Cuisines'].split(',') if (b in popular_cuisines and len(b)>=1)],axis=1)
from functools import reduce
#dynamically populate the indicator columns,for all the cusines that match the cuisines in top list,indicate that with a flag
[ exec(a) for a in reduce(lambda x,y:x+y,Restaurants.apply(lambda a:["Restaurants.loc["+str(a.name)+",'"+b+"']=1" for b in list({b.strip().replace(" ","_") for b in a['Cuisines'].split(",") }.intersection(set(popular_cuisines)))],axis=1).to_list())]
[ exec(a) for a in reduce(lambda x,y:x+y,Restaurants.apply(lambda a:["Restaurants.loc["+str(a.name)+",'"+'least_popular_cuisine'+"']=1" for b in list({b.strip().replace(" ","_") for b in a['Cuisines'].split(",") }.intersection(set(least_popular_cuisines)))],axis=1).to_list())]
Restaurants['Collections'].fillna("",inplace=True)
top_5_collections
#mark the restaurants that are featured in popular collections
[ exec(a) for a in reduce(lambda x,y:x+y,
Restaurants.apply(lambda a:["Restaurants.loc["+str(a.name)+",'"+'featured_in_pop_collections'+"']=1" 
          for b in list({b.strip() 
                         for b in a['Collections'].split(",") }.intersection(set(top_5_collections))
                       )
         ],axis=1).to_list())]
Restaurants
Reviews.isnull().sum()/len(Reviews.Restaurant)
#most of te column has 38% as missing rate.so they all have nulls across all these columns.so we can drop the row
Reviews.drop(Reviews[Reviews.Reviewer.isnull()].index,axis=0,inplace=True)
Reviews.isnull().sum()/len(Reviews.Restaurant)
Reviews['Metadata'].fillna("0 Review , 0 Followers",inplace=True)
#extract the number of reviewers and followers from metadata column
Reviews['no_of_reviews_follwers']=Reviews.Metadata.str.split(",").apply(lambda a:(a[0].strip().split(" ")[0], a[1].strip().split(" ")[0])                                      
                                      if (len(a)==2 and a!="")  else (
                                      
                                      (a[0].strip().split(" ")[0] if a[0].strip().split(" ")[1] in ['Review','Reviews'] else 0,a[0].strip().split(" ")[0] if a[0].strip().split(" ")[1] in ['Follower','Followers'] else 0)
                                       
                                          if (len(a)==1 ) else 0
                                          
                                      )
                                      
                                     )
#create 2 new columns for no of reviews and followers
Reviews['no_of_reviews']=Reviews['no_of_reviews_follwers'].apply(lambda a:a[0])
Reviews['no_of_followers']=Reviews['no_of_reviews_follwers'].apply(lambda a:a[1])
Reviews.head(10)
Reviews['Date']=pd.to_datetime(Reviews.Time)
Reviews[Reviews.Date.isnull()]
#create a new column to store week of month
import math
Reviews['Week_of_month']=pd.to_numeric(Reviews.Date.dt.day/7).apply(lambda a:math.ceil(a))
boundaries=[0,4,8,12,16,20,24]
labels=['early_morning','morning','post_morning','noon','evening','night']
Reviews['Time_of_day']=pd.cut(Reviews.Date.dt.hour,bins=boundaries,labels=labels,include_lowest=True)
#Reviews['Time_of_day']=Reviews.Date.dt.hour
Reviews.drop(Reviews[Reviews['Rating']=='Like'].index,inplace=True)
Reviews['Rating']=Reviews['Rating'].astype(np.number)
Reviews
#eda
import seaborn as sns
sns.boxplot(hue='Time_of_day',y='Rating',data=Reviews,x='Week_of_month')
#merge both restaurant and review data sets
restaurant_reviews=pd.merge(Reviews,Restaurants,left_on='Restaurant',right_on='Name')
