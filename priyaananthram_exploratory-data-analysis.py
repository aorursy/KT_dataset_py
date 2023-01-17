# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import re

from nltk.stem import WordNetLemmatizer



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import pandas as pd

from pylab import rcParams

rcParams['figure.figsize'] = 15, 5

%matplotlib inline

train=pd.read_json("../input/train.json")

test=pd.read_json("../input/test.json")

train['sod']='train'

test['sod']='test'



data=train.append(test)

data['num_features']=data.features.apply(lambda x:len(x))

data['no_of_photos']=data.photos.apply(lambda x:len(x))

data['len_description']=data.description.apply(lambda x:len(x))

data['len_description']=data.len_description.fillna(0)



data['description_word_count']=data.description.apply(lambda x:len(x.split(" ")))



# convert the created column to datetime object so as to extract more features 

data["created"] = pd.to_datetime(data["created"])



# Let us extract some features like year, month, day, hour from date columns #

data["created_year"] = data["created"].dt.year

data["created_month"] = data["created"].dt.month

data["created_day"] = data["created"].dt.day

data["created_hour"] = data["created"].dt.hour





#train['features_string']=[' , '.join(z).lower().strip() for z in train['features']] 

data['features_string']=[','.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)).strip().lower() for line in lists]).strip() for lists in data['features']]

data.features_string=data.features_string.str.replace(" ","_")

data.features_string[0:5]

train=data[data.sod=='train']

test=data[data.sod=='test']



bins1=[0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300]

import numpy as np

data['desc_bins']=np.digitize(data.description_word_count, bins1)



data.head()
print("train ",train.shape)

print("test ",test.shape)

print("total ",data.shape)
data['desc_bins'].value_counts().plot(kind='bar')


ax=train.bathrooms.value_counts().plot(kind='bar')

ax.set(xlabel='bathrooms',ylabel='count',title='bathroom distribution')
train.loc[train.bathrooms>3,'bathrooms']=3



ax=pd.crosstab(train.bathrooms,train.interest_level).plot(kind='bar')

ax.set(ylabel='count',title='number of bathrooms by interest level')
train.bedrooms.value_counts().plot(kind='bar')
train.loc[train.bedrooms>=4,'bedrooms']=4

pd.crosstab(train.bedrooms,train.interest_level).plot(kind='bar')
train.price.quantile([.1, .25,.5,0.6,0.75,0.9,0.95,0.99,1])
train1=train.loc[train.price<13000]

train1.boxplot(by='interest_level',column='price')
train1=train[train.no_of_photos<=14]

pd.crosstab(train1.no_of_photos,train1.interest_level).plot(kind='bar')
train.loc[train.num_features>=17,'num_features']=17

pd.crosstab(train.num_features,train.interest_level).plot(kind='bar')
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

plt.figure(figsize=(15,8))

m = Basemap(projection='merc',llcrnrlat=40.55,urcrnrlat=40.82,area_thresh = 0.1,

            llcrnrlon=-74.1, urcrnrlon=-73.82, lat_ts=40.5,resolution='h')

m.drawcoastlines()

m.drawrivers()

m.fillcontinents()

m.drawmapboundary() 

colors = ['yellow','green','red']



df=train.loc[train.interest_level=='low',:]



x, y = m(list(df["longitude"].astype(float)), list(df["latitude"].astype(float)))

plot1=m.plot(x, y, 'go', markersize = 3, alpha = 0.8, color = colors[0],label='Low')



df=train.loc[train.interest_level=='medium',:]

x, y = m(list(df["longitude"].astype(float)), list(df["latitude"].astype(float)))

plot2=m.plot(x, y, 'go', markersize = 2, alpha = 0.8, color = colors[1],label='Medium')





df=train.loc[train.interest_level=='high',:]

x, y = m(list(df["longitude"].astype(float)), list(df["latitude"].astype(float)))

plot3=m.plot(x, y, 'go', markersize = 1, alpha = 0.8, color = colors[2],label='High')







plt.title('Rental analysis')

plt.legend()

plt.show()

#lower manhattan and upper east side is expensive 
rcParams['figure.figsize'] = 10, 8



plot1=train.loc[train.interest_level=='high'].plot.scatter(x='latitude',y='longitude',c='red').set_title("High interest level")

rcParams['figure.figsize'] = 10, 5



df=train.display_address.value_counts()

ax=df.hist(bins=100,log=True)

ax.set(ylabel="log_count",xlabel='number of times address appears')
df=train.building_id.value_counts()



df.quantile([.1, .25,.5,0.6,0.75,0.9,0.95,0.99,1])

#Seems that one building keeps coming up again and again 0 lets remove this

df[1:].hist(log=True,bins=100).set(title="BuildingID count distribution",xlabel='Number of times buildingId appears',ylabel='log_count')
#Are certain strees preferred
train.street_address.value_counts().hist(log=True,bins=100).set(title='Street address count distribution',xlabel='Number of times street address appears',ylabel='log_count')
from wordcloud import WordCloud

import matplotlib.pyplot as plt



plt.figure(figsize=(12,6))

str1 = ' '.join(data.features_string.tolist())

wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=40,max_words=100).generate(str1)

wordcloud.recolor(random_state=0)

plt.imshow(wordcloud)

plt.title("Wordcloud for features", fontsize=30)

plt.axis("off")

plt.show()



from wordcloud import WordCloud

plt.figure(figsize=(12,6))

str2 = ' '.join(data.description.tolist())

wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=40,max_words=100).generate(str2)

wordcloud.recolor(random_state=0)

plt.imshow(wordcloud)

plt.title("Wordcloud for description", fontsize=30)

plt.axis("off")

plt.show()



from wordcloud import WordCloud

plt.figure(figsize=(12,6))

str2 = ' '.join(data.display_address.tolist())

wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=40,max_words=60).generate(str2)

wordcloud.recolor(random_state=0)

plt.imshow(wordcloud)

plt.title("Wordcloud for display addresses", fontsize=30)

plt.axis("off")

plt.show()



from wordcloud import WordCloud

plt.figure(figsize=(12,6))

str2 = ' '.join(data.street_address.tolist())

wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=40,max_words=60).generate(str2)

wordcloud.recolor(random_state=0)

plt.imshow(wordcloud)

plt.title("Wordcloud for street addresses", fontsize=30)

plt.axis("off")

plt.show()


