# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data_playstore = pd.read_csv('../input/googleplaystore.csv')
data_ps_userreviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')
data_playstore.head()
data_playstore.info()
data_playstore.shape
data_playstore.columns
data_playstore['Category'].unique()
data_playstore.Rating = data_playstore.Rating.astype(float)
# I changed the type for using
data_playstore.Reviews.value_counts()
data_playstore.Reviews.replace(['3.0M'],3.0,inplace = True) #I saw 3.0M data when i look the values so i had to change for numeric values
data_playstore.Reviews = data_playstore.Reviews.astype(float)
data_playstore.plot(kind='scatter', x='Rating', y='Reviews',alpha = 0.5,color = 'red')
#plt.scatter(data.Rating,data.Reviews,alpha = 0.5,color = 'red') 
plt.xlabel('Rating')              # label = name of label
plt.ylabel('Reviews')
plt.title('Rating Reviews Scatter Plot')            # title = title of plot
category_list = list(data_playstore['Category'].unique()) # I append the unique categories in category list
Reviews_ratio = [] #I made a empty list
for i in category_list: 
    x = data_playstore[data_playstore['Category']==i] #Find unique states one by one
    category_Reviews_rate = sum(x.Reviews)/len(x) #average category_Reviews_rate (divide by length)
    Reviews_ratio.append(category_Reviews_rate) #append the empty list
data = pd.DataFrame({'category_list': category_list,'Reviews_ratio':Reviews_ratio}) #I have created data for their names in the category_list and Reviews_ratio
new_index = (data['Reviews_ratio'].sort_values(ascending=False)).index.values #sorted
sorted_data = data.reindex(new_index) #I sort indexed data back to sorted_data

# visualization
plt.figure(figsize=(15,10)) #open a new figure
sns.barplot(x=sorted_data['category_list'], y=sorted_data['Reviews_ratio']) #sns: seaborn library
#x=sorted_data's reviewsrate, y=sorted_data's category
plt.xticks(rotation= 45) 
plt.xlabel('Category') 
plt.ylabel('Reviews_rate')
plt.title('Reviews - Category')
data_playstore.Rating.plot(kind = 'hist',bins = 50,figsize = (12,12))
#Plot the speed of the data. Our species is a histogram. (x: velocity values)
plt.show()
a = data_playstore['Reviews']>38000000     # There are most popular apps
data_playstore[a]
print(data_playstore['Price'].value_counts(dropna =False))
data_playstore.describe()