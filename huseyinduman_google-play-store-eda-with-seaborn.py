# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/googleplaystore.csv')
data.head()
data.tail()
data.info()
#there is value '3.0M' which should not be in reviews column so we drop this column
data.drop(data.Reviews[data.Reviews == '3.0M'].index, inplace = True) 
#replace space with _ so there is no gap between words in column names
data.columns = data.columns.str.replace(' ','_')
#make Reviews number
data.Reviews = pd.to_numeric(data.Reviews ,downcast='integer')
#make Last update date time format
data.Last_Updated= pd.to_datetime(data.Last_Updated)
#add new column called size_number. Sizes are in kilobytes
data['Size_number'] = data.Size.str.replace('M','000')
data.Size_number = data.Size_number.str.replace('k','')
data.Size_number = data.Size_number.replace('Varies with device', np.NaN)
data.Size_number = pd.to_numeric(data.Size_number, downcast = 'integer')
#make installs number
data.Installs = data.Installs.str.replace('+','')
data.Installs = data.Installs.str.replace(',','')
data.Installs = pd.to_numeric(data.Installs,downcast = 'integer')
#make price float
data.Price = data.Price.str.replace('$','')
data.Price = pd.to_numeric(data.Price)
data.info()
data.head()
sns.heatmap(data.corr(),vmax = 1, vmin = 0,annot = True)
plt.show()
#Bar plot of average size of applications by categories
average_size = []
for each in data.Category.unique():
    average_size.append(data[data.Category == each].Size_number.mean()/1000)

average_size_df = pd.DataFrame({'category':data.Category.unique(),'average_size':average_size})
new_index = (average_size_df.average_size.sort_values(ascending = False)).index.values
average_size_df = average_size_df.reindex(new_index)

plt.figure(figsize = (15,10))
sns.barplot(x = average_size_df.category, y = average_size_df.average_size)
plt.xticks(rotation = 90)
plt.xlabel('Category')
plt.ylabel('Size (Mbytes)')
plt.title('Average Size by Categories')
plt.show()
#Bar plot of number of paid and free app by categories 
paid = []
number = []
for each in data.Category.unique():
    paid.append(data[(data.Category == each) & (data.Type == 'Paid')].App.count())
    number.append(data[(data.Category == each)].App.count())

app_df = pd.DataFrame({'Category':data.Category.unique(), 'total':number, 'paid': paid})
new_index = (app_df.total.sort_values(ascending = False)).index.values
app_df =app_df.reindex(new_index)

plt.figure(figsize=(10,15))
sns.barplot(x = app_df.total,y = app_df.Category, alpha = 0.5, label = 'Total', color = 'green')
sns.barplot(x = app_df.paid, y = app_df.Category, alpha = 0.5, label = 'Paid', color = 'red')
plt.xlabel('Number')
plt.ylabel('Category')
plt.title('Total Number of Apps by Category ')
plt.legend()
plt.show()
#point plot of ratings and reviews
avg_rating=[]
avg_review=[]
for each in data.Category.unique():
    avg_rating.append(data[data.Category == each].Rating.mean())
    avg_review.append(data[data.Category == each].Reviews.mean())
    
avg_rating = np.asarray(avg_rating)
avg_review = np.asarray(avg_review)
avg_rating = avg_rating/max(avg_rating)
avg_review = avg_review/max(avg_review)
data_rr = pd.DataFrame({'category':data.Category.unique(),'rating': avg_rating, 'review': avg_review})
new_index = (data_rr.rating.sort_values(ascending = False).index.values)
data_rr = data_rr.reindex(new_index)
plt.figure(figsize=(15,10))
sns.pointplot(x=data_rr.category,y=data_rr.review, alpha = 0.5, color = 'red')
sns.pointplot(x=data_rr.category,y=data_rr.rating, alpha = 0.5, color = 'lime')
plt.xticks(rotation = 90)
plt.xlabel('Category')
plt.ylabel('Normalized Rating and Review')
plt.title('Ratings and Reviews')
plt.show()
#violin plot of ratings by categories

plt.figure(figsize=(15,10))
sns.violinplot(x = data.Category, y = data.Rating)
plt.xticks(rotation= 90)
plt.show()
#countplot of genres of game apps
plt.figure(figsize=(15,10))
sns.countplot(data[data.Category == 'GAME'].Genres)
plt.xticks(rotation = 90)
plt.show()
#count plot of Android version
plt.figure(figsize = (15,10))
sns.countplot(data.Android_Ver)
plt.xticks(rotation = 90)
plt.show()
#swarm plot of ratings of paid and free types of health-fitness and food-drink apps

data_swarm = data[(data.Category == 'HEALTH_AND_FITNESS')|(data.Category == 'FOOD_AND_DRINK')]

plt.figure(figsize = (15,10))
sns.swarmplot(x = data_swarm.Category, y = data_swarm.Rating, hue = data_swarm.Type)
plt.show()
#lm plot of max. normalized reviews and size_number
rev = np.asarray(data.Reviews)
siz = np.asarray(data.Size_number)
rev_n = rev/max(rev)
siz_n = siz/max(siz)
data_dum = pd.DataFrame({'rev':rev_n,'siz':siz_n})
plt.figure()
sns.lmplot(x = 'rev', y= 'siz', data = data_dum)
plt.show()
#kde plot of price and rating of paid apps

data_kde = data[data.Type == 'Paid']
plt.figure(figsize = (15,10))
sns.lmplot( x = 'Rating', y = 'Price' , data = data_kde)
plt.show()
#box plot of ratings of finance and business apps

data_box = data[(data.Category == 'FINANCE') | (data.Category == 'BUSINESS')]
plt.figure(figsize = (15,10))
sns.boxplot(x = data_box.Category, y = data_box.Rating)
plt.show()
#boxenplot of size_number of apps by android version
plt.figure(figsize = (15,10))
sns.boxenplot(x = data.Android_Ver, y = data.Size_number)
plt.xticks(rotation = 90)
plt.show()