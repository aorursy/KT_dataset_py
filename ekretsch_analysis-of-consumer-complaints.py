import numpy as np                          #arrays/computing

import pandas as pd                         #data pre-processing and analysis

import matplotlib.pyplot as plt             #plotting

import seaborn as sns                       #statistical graphing

import re                                   #regular expressions

from os import path                         #os dependent functionality

from wordcloud import WordCloud, STOPWORDS  #wordcloud

from nltk.corpus import stopwords           #natural language toolkit

from PIL import Image                       #image loader



df = pd.read_csv('../input/comcastcomplaints/comcast_consumeraffairs_complaints.csv')



df.head()
df.tail()
#Display size of dataset

print(df.shape)
#Visualize rating data

sns.countplot(data=df,x='rating',palette='cubehelix')
#Sum total reviews

Positive_reviews = int(df['rating'].loc[df['rating'] >= 3].count())

Negative_reviews = int(df['rating'].loc[df['rating'] < 3].count())

print('There are a total of ' + str(Positive_reviews) + ' positive reviews and a total of ' + str(Negative_reviews) + ' negative reivews.')
#Clean data and re-plot

df = df.loc[df.rating < 3]

sns.countplot(data=df,x='rating',palette='cubehelix')
print('There are ' + str(Negative_reviews) + ' total negative reviews.')
#Visualize the location data

df['state'] = df['author'].str.extract(r'(AL|Alabama|AK|Alaska|AZ|Arizona|AR|Arkansas|CA|California|CO|Colorado|CT|Connecticut|DE|Delaware|FL|Florida|GA|Georgia|HI|Hawaii|ID|Idaho|IL|Illinois|IN|Indiana|IA|Iowa|KS|Kansas|KY|Kentucky|LA|Louisiana|ME|Maine|MD|Maryland|MA|Massachusetts|MI|Michigan|MN|Minnesota|MS|Mississippi|MO|Missouri|MT|Montana|NE|Nebraska|NV|Nevada|NH|New Hampshire|NJ|New Jersey|NM|New Mexico|NY|New York|NC|North Carolina|ND|North Dakota|OH|Ohio|OK|Oklahoma|OR|Oregon|PA|Pennsylvania|RI|Rhode Island|SC|South Carolina|SD|South Dakota|TN|Tennessee|TX|Texas|UT|Utah|VT|Vermont|VA|Virginia|WA|Washington|WV|West Virginia|WI|Wisconsin|WY|Wyoming)')

df.state.value_counts()
# Top 10 states with the most complaints

topTen = pd.value_counts(df['state'].values, sort=True).head(10)

topTen.plot.barh(title="Top 10 Complaints by State", x="Number of complaints", y="States")
# Extract year from 'posted_on' column

df['year'] = df['posted_on'].str.extract(r'([1-3][0-9]{3})')

df.year.value_counts()
#Plot complaints as timeseries

timeseries = pd.DataFrame({

        'complaints': [34, 106, 441, 460, 415, 347, 409, 309, 754, 1452, 807]

    }, index=[2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016])

lines = timeseries.plot.line()
#Create wordcloud out of 'text' column



#Join complaints from all rows in text column

text = " ".join(str(complaint) for complaint in df.text)

print ("There are {} words in the combination of all complaints.".format(len(text)))



#create stopword list:

stopwords = set(STOPWORDS)

stopwords.update(["Comcast", "even", "will", "now", "lot", "wow", "xfinity", "us", "want", "going", "told"])

wordcloud = WordCloud(stopwords=stopwords, max_words=100, width=4000, height=2000, background_color="black").generate(text)



#show plot

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()