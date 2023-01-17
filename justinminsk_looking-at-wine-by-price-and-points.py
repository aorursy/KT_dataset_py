import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS

#import libraries
wine = pd.read_csv('../input/winemag-data_first150k.csv')

wine.head()

#import data
wine = wine.dropna()

#get rid of rows with na
sns.distplot(wine['points'], kde=False);

#looking at the point distribution 
sns.distplot(wine['price'], kde=False);

#looking at distribution of price
sns.regplot(x='points', y='price', x_estimator=np.mean, data=wine)

#shows price and points on the same scale
points_by_price = wine['points'] / wine['price']

#create points divided by price

s1 = pd.Series(points_by_price, name = 'Points by Price')

#create a series with points divided by price

wine2 = pd.concat([wine, s1], axis = 1)

#add points by price to wine data
sns.distplot(wine2['Points by Price'], kde=False);

#look at distribution of points divided by price
US = wine['country'].map(lambda x: str(x).startswith('US'))

#get a boolean list with true being us in country

USwine = wine2[US]

#make a list of just us wines

USwine = USwine.dropna()

#drop na rows

sns.distplot(USwine['Points by Price'], kde=False);

#look at distribution of wines
sorted_US = USwine.sort_values(by = ['Points by Price'], ascending=False)

#look at the list sorted by points by price

sorted_US.head(10)

#look at top ten
GT80 = USwine['points'].map(lambda x: x > 89)

#only look at points 90 or above

GT80USwine = USwine[GT80]

#turn it into a series

sorted_US = GT80USwine.sort_values(by = ['Points by Price'], ascending=False)

#sort by points by price

sorted_US.head(10)

#look at top ten
GT90 = USwine['points'].map(lambda x: x > 94)

#look at points 95 and over

GT90USwine = USwine[GT90]

#make a list of 95 and over point wines

sorted_US = GT90USwine.sort_values(by = ['Points by Price'], ascending=False)

#sort it

sorted_US.head(10)

#look at top ten
mpl.rcParams['figure.figsize']=(7.0,6.0)

mpl.rcParams['font.size']=12                

mpl.rcParams['savefig.dpi']=100              

mpl.rcParams['figure.subplot.bottom']=.1 

#make the frame the wordcloud will be in



stopwords = set(STOPWORDS)

data = USwine

#get stop words and the data



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=100,

                          max_font_size=50, 

                          random_state=False

                         ).generate(str(data['description']))

#varibles for the wordcloud



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

#show wordcloud
