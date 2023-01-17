#import library

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud

#read to csv

data = pd.read_csv("../input/googleplaystore.csv")
data.info()
data.columns
data.shape
data.head()
data.tail()
data1 = data.head()

data2 = data.tail()

concat_data = pd.concat([data1,data2],axis=0,ignore_index=True)

concat_data
data['Category'].unique()
data[data['Category'] == '1.9']
data.loc[10472] = data.loc[10472].shift()

data['App'].loc[10472] = data['Category'].loc[10472]

data['Category'].loc[10472] = np.nan

data.loc[10472]
data['Rating'].unique()
data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')

data['Rating'].dtype
data['Reviews'].unique()
data[data['Reviews'] =='3.0M']
data['Reviews'] = data.Reviews.replace("0.0",0)

data['Reviews'] = data.Reviews.replace("3.0M",3000000.0)

data['Reviews'] = data['Reviews'].astype(float)

data['Reviews'].dtype
data['Size'].unique()
data['Size'] = data.Size.replace("Varies with device",np.nan)

data['Size'] = data.Size.str.replace("M","000") # All size values became the kilobyte type.

data['Size'] = data.Size.str.replace("k","")

data['Size'] = data.Size.replace("1,000+",1000)

data['Size'] =data['Size'].astype(float)

data['Size'].dtype
data['Installs'].unique()
data['Installs'] = data.Installs.str.replace(",","")

data['Installs'] = data.Installs.str.replace("+","")

data['Installs'] = data.Installs.replace("Free",np.nan)

data['Installs'] = data['Installs'].astype(float)

data['Installs'].dtype
data['Price'].unique()
data['Price'] = data.Price.replace("Everyone",np.nan)

data['Price'] = data.Price.str.replace("$","").astype(float)

data['Price'].dtype
data['Last Updated'].unique()
data['Last Updated'] = pd.to_datetime(data['Last Updated'])

data['Last Updated']
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.describe()
category_list = list(data['Category'].unique())

category_review = []

for i in category_list:

    x = data[data['Category'] == i]

    if(len(x)!=0):

        review = sum(x.Reviews)/len(x)

        category_review.append(review)

    else:

        review = sum(x.Reviews)

        category_review.append(review)

#sorting

data_category_reviews = pd.DataFrame({'category': category_list,'review':category_review})

new_index = (data_category_reviews['review'].sort_values(ascending=False)).index.values

sorted_data =data_category_reviews.reindex(new_index)



# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['category'], y=sorted_data['review'])

plt.xticks(rotation=80)

plt.xlabel("Category")

plt.ylabel("Reviews")

plt.title("Category and Reviews")

plt.show()
category_list = list(data['Category'].unique())

category_install = []

for i in category_list:

    x = data[data['Category'] == i]

    if(len(x)!=0):

        install = sum(x.Installs)/len(x)

        category_install.append(install)

    else:

        install = sum(x.Installs)

        category_install.append(install)

        

#sorting

data_category_install = pd.DataFrame({'category': category_list,'install':category_install})

new_index = (data_category_install['install'].sort_values(ascending=False)).index.values

sorted_data =data_category_install.reindex(new_index)



# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['category'], y=sorted_data['install'])

plt.xticks(rotation=80)

plt.xlabel("Category")

plt.ylabel("Install")

plt.title("Category and Install")

plt.show()
plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(data))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
plt.figure(figsize=(10,7))

sns.countplot(data=data, x='Content Rating')

plt.xticks(rotation=80)

plt.title('Content Rating',color = 'blue',fontsize=15)

plt.show()