# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt 



#plotly

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()

import plotly.graph_objs as py



#for word cloud

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



import re

import os

print(os.listdir("../input"))

from IPython.display import HTML

from IPython.display import display



from PIL import Image

from termcolor import colored



from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','']) # remove it if you need punctuation 



from nltk.stem import WordNetLemmatizer



import seaborn as sns

import nltk



#to supress Warnings 

import warnings

warnings.filterwarnings("ignore")





#---------------------------------------------

import pandas as pd

import string

from nltk import word_tokenize

from nltk.corpus import stopwords

from textblob import TextBlob # for sentiment analysis

from collections import Counter 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/world-cup-2018-tweets/FIFA.csv')

df.head(5)
display(df.info())

#dropping null tweets

df.dropna(subset=['Tweet'], inplace=True)

display(df.info())
#replacing one or multiple occurences of '?' in name (cleaning name)

df.Name = df.Name.str.replace('[\?]+','unknown', regex = True) 

df.Name
df.groupby('Place')['Name'].count().sort_values(ascending=False)
#finding place of one of the most influence tweet (RTs)

kt = df[['UserMentionID','Followers','Place','Orig_Tweet','Tweet', 'RTs']].sort_values(by = 'RTs', ascending = False).head(1)

display(kt.head(1))
df.Tweet[529999]
# Clean and Normalize Text

# - tokenize

# - lowercase

# - remove punctuation

# - remove alphanumeric characters

# - remove stopwords



stopwords = set(stopwords.words('english'))



def clean(text):

    text = word_tokenize(text)

    text = [word.lower() for word in text]

    punct = str.maketrans('', '', string.punctuation) 

    text = [word.translate(punct) for word in text] 

    text = [word for word in text if word.isalpha()]

    text = [word for word in text if not word in stopwords]

    return " ".join(text)



df['clean_tweet'] = df['Tweet'].apply(clean)

df.head(5)



# Create Word Count Column for Clean Text



df['clean_word_count'] = df['clean_tweet'].str.split().str.len()

df.head(5)
#tokenizing tweets

text = ' '.join(df.clean_tweet)

display(text)

text = word_tokenize(text)

text = [word.lower() for word in text]

punct = str.maketrans('', '', string.punctuation) 

text = [word.translate(punct) for word in text] 

text = [word for word in text if word.isalpha()]

text = [word for word in text if not word in stopwords]

print(text)
tags = nltk.pos_tag(text)

nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')

]



#series with nouns

s_index = pd.Series(nouns)



print(nouns)


#Word cloud for Fifa -- lets see some popular Key words from the tweets 

wave_mask = np.array(Image.open( "../input/beerimage/fifacup_.png"))

wordcloud = (WordCloud(width=1440, height=1080, mask = wave_mask, relative_scaling=0.5, stopwords=stopwords, background_color='grey').generate_from_frequencies(s_index.value_counts()))





fig = plt.figure(1,figsize=(15, 15))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
# Apply Sentiment Polarity to Text with TextBlob



df['polarity'] = [round(TextBlob(word).sentiment.polarity, 2) for word in df['clean_tweet']]

df['sentiment'] = ['positive' if polarity > 0 

                             else 'negative' if polarity < 0 

                                 else 'neutral' 

                                     for polarity in df['polarity']]



#Sentiments

df.sentiment.value_counts().plot(kind='pie')
#Can we predict RTs from followers, friends, hastag_count,sentiment, word_count?



#hastag count - numeric fields required

#df['hash_count'] = df.Hashtags.map(lambda x: [i.strip() for i in x.split(",")])

df['hash_count'] = df.Hashtags.apply(lambda x : len(str(x).split(',')))

dff = df[['RTs','hash_count','Followers','Friends','len']]

dff.head
#Pairplot 

import seaborn as sns



sns.set(style="ticks", color_codes=True)

#iris = sns.load_dataset("FIFA'18 Tweets")

g = sns.pairplot(dff)





import matplotlib.pyplot as plt

plt.show()
#predict RTs (Popularity of Tweet - what makes a tweet tweetable?)

from sklearn.model_selection import train_test_split



X = dff[['hash_count','Followers','Friends','len']]

y = dff['RTs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# Feature Scaling

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



from sklearn.ensemble import RandomForestRegressor



regressor = RandomForestRegressor(n_estimators=20, random_state=0)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)



from sklearn import metrics



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))