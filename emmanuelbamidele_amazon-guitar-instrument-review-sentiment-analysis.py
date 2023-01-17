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
import os

import re

import numpy as np

import pandas as pd

from textblob import TextBlob

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid', context='talk', palette='Dark2')

import nltk

from nltk import word_tokenize
data = "/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv"

df = pd.read_csv(data)

df.head()
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA



sid = SIA()

results = []



def get_sentiment(row, **kwargs):

    sentiment_score = sid.polarity_scores(row)

    positive_meter = round((sentiment_score['pos'] * 10), 2)

    negative_meter = round((sentiment_score['neg'] * 10), 2) 

    return positive_meter if kwargs['k'] == 'positive' else negative_meter



df['positive'] = df.summary.apply(get_sentiment, k='positive')

df['negative'] = df.summary.apply(get_sentiment, k='negative')

df['neutral'] = df.summary.apply(get_sentiment, k='neutral')

df['compound'] = df.summary.apply(get_sentiment, k='compound')



#for index, row in df.iterrows(): 

 #  print("Positive : {}, Negative : {}, Neutral : {}, Compound : {}".format(row['positive'], row['negative'], row['neutral'], row['compound']))
df.head()
df['label'] = 0

df.loc[df['positive'] > 5, 'label'] = 1

df.loc[df['negative'] > 3, 'label'] = -1

df.head()
#save new data for easy use

df.rename(columns={'overall':'overallRating'}, inplace=True)

df2 = df[['reviewerName', 'reviewText', 'overallRating', 'summary', 'reviewTime', 'positive', 'negative', 'neutral', 'label']]

df2.to_csv('Amazon_musical_instrument_review.csv', mode='a', encoding='utf-8', index=False)

df2.head()
#Checking how many positives and negatives we have in the data

#The first line gives us raw value counts of the labels, 

#whereas the second line provides percentages with the normalize keyword.



print(df2.label.value_counts())



print(df2.label.value_counts(normalize=True) * 100)
#Plotting a bar chart



sns.set(rc={'figure.figsize':(8,6)})



counts = df2.label.value_counts(normalize=True) * 100



ax = sns.barplot(x=counts.index, y=counts)

ax.set(title="Plot of Percentage Sentiment");

ax.set_xticklabels(['Negative', 'Neutral', 'Positive']);

ax.set_ylabel("Percentage");
#boxplot to see average values of the labels and the postivity





boxplot = df2.boxplot(column=['positive','label'], 

                     fontsize = 15,grid = True, vert=True,figsize=(8,5,))

plt.ylabel('Range');
#Classify Ratings based on high or low



def convert_rating(rating_values):

    if(int(rating_values == 1) or int(rating_values) == 2 or int(rating_values) == 3):

        return 0

    else:

        return 1

   

df2["Ratings_classified"] = df2["overallRating"];

df2.Ratings_classified = df2.Ratings_classified.apply(convert_rating)
df2.head()

df2.overallRating.value_counts()

sns.set(rc={'figure.figsize':(8,6)})

ax = sns.countplot(x = 'Ratings_classified' , hue = 'Ratings_classified' , data = df2).set(title="Classification of Ratings based on high or low", xlabel="Ratings", ylabel="Quantity")

plt.legend(["Low (<=3)", "High(>3)"])

plt.show()
#Ratings Distribution for dataset



plt.hist(df2['overallRating'], color = 'darkblue', edgecolor = 'black', density=False,

         bins = int(30))

plt.title('Ratings Distribution');

plt.xlabel("Ratings");

plt.ylabel("Number of TImes");



from pylab import rcParams

rcParams['figure.figsize'] = 8,8
#Placing a density curve on the distribution



sns.distplot(df2['overallRating'], hist=True, kde=True, 

             bins=int(30), color = 'darkred',

             hist_kws={'edgecolor':'black'},axlabel ='Ratings')

plt.title('Ratings Density')



from pylab import rcParams

rcParams['figure.figsize'] = 8,8
#nltk.download()



stopwords = nltk.corpus.stopwords.words('english')

RE_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))

words = (df.summary

           .str.lower()

           .replace([r'\|',r'\&',r'\-',r'\.',r'\,',r'\'', RE_stopwords], [' ', '','','','','',''], regex=True)

           .str.cat(sep=' ')

           .split()

)
from collections import Counter



# generate DF out of Counter

rslt = pd.DataFrame(Counter(words).most_common(10),

                    columns=['Word', 'Frequency']).set_index('Word')

rslt
rslt_wordcloud = pd.DataFrame(Counter(words).most_common(100),

                    columns=['Word', 'Frequency'])

#BAR CHART

rslt.plot.bar(rot=40, figsize=(10,6), width=0.8,colormap='tab10')

plt.title("Commonly used words by Buyers of the Amazon Musical Instrument")

plt.ylabel("Count")



from pylab import rcParams

rcParams['figure.figsize'] = 8,6
#PIE CHART



explode = (0.1, 0.12, 0.122, 0,0,0,0,0,0,0)  # explode 1st slice

labels=['great',

        'good',

        'nice',

        'guitar',

        'works',

        'price',

        'strings',

        'best',

        'quality',

        'stand',]



plt.pie(rslt['Frequency'], explode=explode,labels =labels , autopct='%1.1f%%',

        shadow=False, startangle=90)

plt.legend( labels, loc='best',fontsize='x-small',markerfirst = True)

plt.tight_layout()

plt.title("Commonly used words by Buyers of the Amazon Musical Instrument")

plt.show()



import matplotlib as mpl

mpl.rcParams['font.size'] = 10
from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import random



wordcloud = WordCloud(max_font_size=60, max_words=100, width=480, height=380,colormap="brg",

                      background_color="white").generate(' '.join(rslt_wordcloud['Word']));

                      

plt.imshow(wordcloud, interpolation='bilinear');

plt.axis("off");

plt.figure(figsize=[10,10]);

plt.show();