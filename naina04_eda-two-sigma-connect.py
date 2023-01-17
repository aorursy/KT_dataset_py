# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



color=sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment =None



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df =pd.read_json("/kaggle/input/two-sigma-connect-rental-listing-inquiries/train.json")

train_df.head(5)
train_df.describe()
# Print top rows



train_df.head()



#Show complete information about your data 



train_df.info()
#Check for nulls



train_df.isnull().max()
#Check no of rows and columns in the train dataset



train_df.shape



#Check value counts of column bedrooms

train_df.bedrooms.value_counts()
#Check value counts of column Interest Level

train_df.interest_level.value_counts()
#Read test data



test_df =pd.read_json("/kaggle/input/two-sigma-connect-rental-listing-inquiries/test.json")
#Show complete information of your test dataset

test_df.info()
#Show top rows

test_df.head()
#Show last rows

test_df.tail()
#Check for nulls



test_df.isnull().max()
#Check for no of rows and columns for test data set



test_df.shape
#Check value counts of column bedrooms

test_df.bedrooms.value_counts()
# Plot interest level



inter_level= train_df['interest_level'].value_counts()



plt.figure(figsize=(8,4))

sns.barplot(inter_level.index,inter_level.values, alpha=0.8, color="blue")

plt.xlabel("Interest Level by category")

plt.ylabel("No of ocurrances")

plt.title("Interest level chart")
#Plot numerical features 



bathroom_counts=train_df["bathrooms"].value_counts()



plt.figure(figsize=(8,4))

sns.barplot(bathroom_counts.index,bathroom_counts.values,alpha=0.8,color="orange")

plt.xlabel("Bathroom Counts")

plt.ylabel("No of occurances")

plt.title("Bathroom count chart")
# bathroom Counts vs Interest Level



train_df['bathrooms'].loc[train_df["bathrooms"]>3]=3



plt.figure(figsize=(12,8))

sns.violinplot(x='interest_level',y="bathrooms",data=train_df,palette="winter", inner="box")

plt.xlabel("Interest Level")

plt.ylabel("Bathrooms")

plt.title("Bathroom Counts vs Interest Level")

plt.show()
#Bedroom Counts



bedroom_counts=train_df["bedrooms"].value_counts()



plt.figure(figsize=(8,4))

sns.barplot(bedroom_counts.index,bedroom_counts.values,alpha=0.8,color="blue")

plt.xlabel("Bedroom Counts")

plt.ylabel("No of occurances")

plt.title("Bedroom count chart")
# bedroom Counts vs Interest Level



train_df['bathrooms'].loc[train_df["bedrooms"]>3]=3



plt.figure(figsize=(12,8))

sns.violinplot(x='interest_level',y="bedrooms",data=train_df,palette="Set2", inner="box")

plt.xlabel("Interest Level")

plt.ylabel("Bedrooms")

plt.title("Bedroom Counts vs Interest Level")

plt.show()
train_df.bedrooms.value_counts()
plt.figure(figsize=(8,4))

sns.countplot(x='bedrooms', hue='interest_level',data=train_df)

plt.ylabel('Number of Occurrences', fontsize=10)

plt.xlabel('bedrooms', fontsize=10)

plt.title("Bedrooms counts by interest level chart ")

plt.show()
#Price Variable Distribution

plt.figure(figsize=(12,8))

plt.scatter(range(train_df.shape[0]),np.sort(train_df.price.values))

plt.xlabel("Index")

plt.ylabel("Price")

plt.title("Price Variable Distribution")

plt.show()



upper_limit=np.percentile(train_df.price.values,99)

train_df['price'].loc[train_df['price']>upper_limit]=upper_limit



plt.figure(figsize=(12,8))

sns.distplot(train_df.price.values,bins=50,kde=True)

plt.xlabel('price', fontsize=12)

plt.show()
#Latitude and Longitude



lower_limit=np.percentile(train_df.latitude.values,1)

upper_limit=np.percentile(train_df.latitude.values,99)

train_df['latitude'].loc[train_df['latitude']>upper_limit]=upper_limit

train_df['latitude'].loc[train_df['latitude']<lower_limit]=lower_limit



plt.figure(figsize=(12,8))

sns.distplot(train_df.latitude.values,bins=50,kde=True)

plt.xlabel('latitude', fontsize=12)

plt.show()
lower_limit=np.percentile(train_df.longitude.values,1)

upper_limit=np.percentile(train_df.longitude.values,99)

train_df['longitude'].loc[train_df['longitude']>upper_limit]=upper_limit

train_df['longitude'].loc[train_df['longitude']<lower_limit]=lower_limit



plt.figure(figsize=(12,8))

sns.distplot(train_df.longitude.values,bins=50,kde=True)

plt.xlabel('longitude', fontsize=12)

plt.show()
# Take a look at Date created



train_df['created']=pd.to_datetime(train_df['created'])

train_df['created_date']= train_df['created'].dt.date



count_srs= train_df['created_date'].value_counts()



plt.figure(figsize=(10,6))



ax= plt.subplot(111)

ax.bar(count_srs.index,count_srs.values,alpha=0.8)

ax.xaxis_date()

plt.xticks(rotation='vertical')

plt.show()
test_df['created']=pd.to_datetime(test_df['created'])

test_df['created_date']= test_df['created'].dt.date



count_srs= test_df['created_date'].value_counts()



plt.figure(figsize=(10,6))



ax= plt.subplot(111)

ax.bar(count_srs.index,count_srs.values,alpha=0.8)

ax.xaxis_date()

plt.xticks(rotation='vertical')

plt.show()
#Hour Created



train_df['hour_created']=train_df['created'].dt.hour







count_srs= train_df['hour_created'].value_counts()



plt.figure(figsize=(10,6))



plt.figure(figsize=(12,6))

sns.barplot(count_srs.index, count_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel("Hours")

plt.ylabel("No of listings")

plt.show()
#Photos



train_df["n_pics"]=train_df['photos'].apply(len)

count_srs= train_df['n_pics'].value_counts()



plt.figure(figsize=(10,6))

sns.barplot(count_srs.index,count_srs.values,alpha=0.8)

plt.xlabel('Number of Photos', fontsize=10)

plt.ylabel('Number of Occurrences', fontsize=10)

plt.show()
train_df['n_pics'].loc[train_df['n_pics']>12]=12

plt.figure(figsize=(10,6))

sns.violinplot(x='n_pics',y='interest_level', data=train_df,order=['low','medium','high'])

plt.xlabel("No of photos")

plt.ylabel("Interest Level")

plt.title("No of photos vs interest level")

plt.show()
#Features



train_df["n_features"]=train_df['features'].apply(len)

count_srs= train_df['n_features'].value_counts()



plt.figure(figsize=(10,6))

sns.barplot(count_srs.index,count_srs.values,alpha=0.8)

plt.xlabel('Number of Features', fontsize=10)

plt.ylabel('Number of Occurrences', fontsize=10)

plt.show()
train_df['n_features'].loc[train_df['n_features']>17]=17

plt.figure(figsize=(12,10))

sns.violinplot(y='n_features',x='interest_level',orient="v", data=train_df,order=['low','medium','high'])

plt.xlabel("No of Features")

plt.ylabel("Interest Level")

plt.title("No of Features vs interest level")

plt.show()
from wordcloud import WordCloud



t_f=''

t_a=''

#t_desc=''



for ind,row in train_df.iterrows():

    for feature in row['features']:

        t_f = "".join([t_f,"_".join(feature.strip().split(" "))])

    t_a="".join([t_a,"_".join(row['display_address'].strip().split(" "))])

    #t_desc="".join([t_desc,row['description']])

t_f=t_f.strip()

t_a=t_a.strip()

#t_desc=t_desc.strip()



#Wordcloud for features



plt.figure(figsize=(14,6))

wordcloud=WordCloud(background_color='white',width=900,height=400,max_words=40,max_font_size=60).generate(t_f)

wordcloud.recolor(random_state=0)

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Wordcloud for Features")

plt.show()





#Wordcloud for Displayaddress



plt.figure(figsize=(14,6))

wordcloud=WordCloud(background_color='white',width=900,height=400,max_words=40,max_font_size=60).generate(t_a)

wordcloud.recolor(random_state=0)

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Wordcloud for Display Address")

plt.show()








