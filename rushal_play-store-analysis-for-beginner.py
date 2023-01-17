# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_play_store = pd.read_csv("../input/googleplaystore.csv")
data_play_store.shape
data_play_store.head()
data_play_store[data_play_store.App == '10 Best Foods for You']
data_play_store.drop_duplicates(subset='App',inplace=True)
data_play_store = data_play_store[data_play_store["Android Ver"] != np.nan]
data_play_store = data_play_store[data_play_store['Android Ver'] != 'NaN']
data_play_store = data_play_store[data_play_store.Installs !="Free"]
data_play_store = data_play_store[data_play_store.Installs !="Paid"]
print("Number of Apps in dataset:{}".format(len(data_play_store)))
data_play_store.Installs = data_play_store.Installs.apply(lambda x:x.replace('+',"") if "+" in str(x) else x)
data_play_store.Installs = data_play_store.Installs.apply(lambda x:x.replace(',',"") if "," in str(x) else x)
data_play_store.Installs = data_play_store.Installs.apply(lambda x:int(x))
data_play_store.info()
#Cleaning Size column

data_play_store.Size = data_play_store.Size.apply(lambda x : x.replace("Varies with device",'NaN') if'Varies with device' in str(x) else x)

#Removing M in column
data_play_store.Size = data_play_store.Size.apply(lambda x : x.replace("M",'') if "M" in str(x) else x)

#Converting data in MB form and removing k
data_play_store.Size = data_play_store.Size.apply(lambda x : float(str(x).replace('k',''))/1000 if 'k' in str(x) else x)

#Convertng type to float
data_play_store.Size = data_play_store.Size.apply(lambda x : float(x))

#Converting type to float
data_play_store.Installs = data_play_store.Installs.apply(lambda x : float(x))
data_play_store.Price = data_play_store.Price.apply(lambda x : x.replace("$",'') if '$' in str(x) else x)
data_play_store.Price = data_play_store.Price.astype(float)
data_play_store.Reviews = data_play_store.Reviews.astype(float)
data_play_store.info()
number_of_category = data_play_store.Category.value_counts().sort_values(ascending=False)
explode=(0,0.1,0,0)
plt.pie(x=number_of_category.values,labels=number_of_category.index,autopct='%1.1f%%', shadow=True,startangle=90)
#plt.legend()
plt.tight_layout()
data_play_store.Rating.plot(kind='hist')
print("Average app rating:{}".format(data_play_store.Rating.mean()))
data_play_store[["Category",'App']][data_play_store.Price > 200]
subset_df = data_play_store[data_play_store.Category.isin(['GAME', 'FAMILY', 'PHOTOGRAPHY', 'MEDICAL', 'TOOLS', 'FINANCE',
                                 'LIFESTYLE','BUSINESS'])]
fig,ax = plt.subplots()
fig.set_size_inches(15,8)
subset_data_price = subset_df[subset_df.Price < 100]

sns.stripplot(x="Price",y='Category',data=subset_data_price,linewidth=1)
plt.show(fig)
plt.figure(figsize=(15,8))
fig = sns.countplot(x=data_play_store.Installs,palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation = 90)
plt.show(fig)
plt.figure(figsize=(6,4))
fig = sns.countplot(x=data_play_store["Type"])
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.show(fig)
plt.figure(figsize=(6,3))
fig = sns.countplot(x=data_play_store["Content Rating"],palette='hls')
fig.set_xticklabels(fig.get_xticklabels(),rotation=80)
plt.show(fig)
plt.figure(figsize=(20,5))
fig = sns.countplot(x=data_play_store.Category,palette='hls')
fig.set_xticklabels(fig.get_xticklabels(),rotation=80)
plt.show(fig)
geners = data_play_store.Genres.value_counts()[:10]
plt.figure(figsize=(15,8))
fig = sns.barplot(x=geners.index,y=geners.values)
fig.set_xticklabels(fig.get_xticklabels(),rotation=80)
plt.show(fig)
rating = data_play_store.sort_values(by=['Rating'],ascending=False)[["App",'Rating']][:10]
plt.figure(figsize=(15,7))
fig = sns.barplot(x=rating.App,y=rating.Rating,palette='hls')
fig.set_xticklabels(fig.get_xticklabels(),rotation=80)
plt.tight_layout()
plt.show(fig)
sorted_reviews = data_play_store.sort_values(by=['Reviews'],ascending=False)[["App",'Reviews']][:10]
plt.figure(figsize=(14,7))
fig = sns.barplot(x=sorted_reviews.App,y=sorted_reviews.Reviews)
fig.set_xticklabels(fig.get_xticklabels(),rotation=80)
plt.tight_layout()
plt.show(fig)
sorted_price = data_play_store.sort_values(by="Price",ascending=False)[["App","Price"]][:10]
plt.figure(figsize=(14,7))
fig = sns.barplot(x=sorted_price.App,y=sorted_price.Price)
fig.set_xticklabels(fig.get_xticklabels(),rotation=80)
plt.show(fig)
