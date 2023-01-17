import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





import numpy as np

import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=20)     

plt.rc('ytick', labelsize=20)



from wordcloud import WordCloud, STOPWORDS

from PIL import Image

import urllib

import requests

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=20)     

plt.rc('ytick', labelsize=20)



from wordcloud import WordCloud, STOPWORDS

from PIL import Image

import urllib

import requests

import pandas as pd
data = pd.read_csv('/kaggle/input/zomato-restaurants-in-pune/zomato_restaurants_in_India.csv')

data=data[data.city=="Pune"]

data=data.drop(['res_id','url','address','latitude','longitude','country_id','zipcode','city','city_id','locality_verbose','currency'],axis=1)

data.index = range(len(data))

data=data.drop_duplicates(subset=None, keep='first')

data.shape

data.head(5)
est_count = data['establishment'].value_counts()

est_count = est_count.sort_values(ascending=True, axis=0)

plt.figure(figsize=(35,25))

plt.xlabel('number of restaurants')

plt.ylabel('establishments')

plt.barh(est_count.index,est_count.values, color='green')
#top 10 locality

loc_count = data['locality'].value_counts() 

loc_count = loc_count.sort_values(ascending=False, axis=0)

loc_count=loc_count.head()

loc_count = loc_count.sort_values(ascending=True, axis=0)

loc_count
plt.figure(figsize=(35,10))

plt.xlabel('number of restaurants')

plt.ylabel('localities')

plt.barh(loc_count.index,loc_count.values, color='blue')
words=data["cuisines"]

words=words.str.cat(sep=', ')

#words
mask = np.array(Image.open("/kaggle/input/applepng/apple.png"))
# This function takes in your text and your mask and generates a wordcloud. 

def generate_wordcloud(words, mask):

    word_cloud = WordCloud(width = 512, height = 512, background_color='black', stopwords=STOPWORDS, mask=mask).generate(words)

    plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')

    plt.imshow(word_cloud)

    plt.axis('off')

    plt.tight_layout(pad=0)

    plt.show()

    

#Run the following to generate your wordcloud

generate_wordcloud(words, mask)
total=data.shape[0]

digital=data[data['highlights'].str.contains('Card|Digital')]

alcohol=data[data['highlights'].str.contains('Alcohol')]

booking=data[data['highlights'].str.contains('book|table')]
dig=(digital.shape[0]/total*100, (total-digital.shape[0])/total*100)

alc=(alcohol.shape[0]/total*100, (total-alcohol.shape[0])/total*100)

book=(booking.shape[0]/total*100, (total-booking.shape[0])/total*100)
N = 3

yes = (digital.shape[0]/total*100,alcohol.shape[0]/total*100   ,booking.shape[0]/total*100)

no = ((total-digital.shape[0])/total*100,(total-alcohol.shape[0])/total*100  , (total-booking.shape[0])/total*100 )
ind = np.arange(N)    

width = 0.30   

plt.figure(figsize=(10,5))

p1 = plt.bar(ind, yes, width)

p2 = plt.bar(ind, no, width, bottom=yes)

plt.ylabel('Percentage')

plt.title('Scores for Restaurants')

plt.xticks(ind, ('Digital', 'Alcohol','Booking'))

plt.yticks(np.arange(0, 101, 10))

plt.legend((p1[0], p2[0]), ('Yes', 'No'))



plt.show()
rating1=data.average_cost_for_two[data.aggregate_rating>4.5]

rating1=rating1.mean(axis=0)



rating2=data.average_cost_for_two[(data.aggregate_rating>4) & (data.aggregate_rating<=4.5)]

rating2=rating2.mean(axis=0)



rating3=data.average_cost_for_two[(data.aggregate_rating>3.5) & (data.aggregate_rating<=4)]

rating3=rating3.mean(axis=0)



rating4=data.average_cost_for_two[(data.aggregate_rating>3) & (data.aggregate_rating<=3.5)]

rating4=rating4.mean(axis=0)



rating5=data.average_cost_for_two[(data.aggregate_rating>2.5) & (data.aggregate_rating<=3)]

rating5=rating5.mean(axis=0)



rating6=data.average_cost_for_two[(data.aggregate_rating>2) & (data.aggregate_rating<=2.5)]

rating6=rating6.mean(axis=0)



rating7=data.average_cost_for_two[(data.aggregate_rating<2)]

rating7=rating7.mean(axis=0)

height = [rating7,rating6,rating5,rating4,rating3,rating2,rating1]

bars = ('1-2','2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-4.5','4.5-5')

y_pos = np.arange(len(bars))



plt.rc('xtick', labelsize=15)     

plt.rc('ytick', labelsize=15)

plt.xlabel('rating')

plt.ylabel('avg cost for two people(rupees)')

plt.bar(y_pos, height, color=['black', 'red', 'green', 'blue', 'cyan','yellow','orange'])

plt.xticks(y_pos, bars)

plt.show()