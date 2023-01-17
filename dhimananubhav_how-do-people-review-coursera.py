import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

sns.set()
%matplotlib inline 

import warnings
warnings.filterwarnings(action='ignore')
df = pd.read_csv('../input/reviews_by_course.csv')
df = df.fillna('')
display(df.info())
display(df.head(10))
%%time
# Number of Words 
df['word_count'] = df['Review'].apply(lambda x: len(str(x).split(" ")))

# Number of characters
df['char_count'] = df['Review'].str.len() ## this also includes spaces

# Number of stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
df['stopwords'] = df['Review'].apply(lambda x: len([x for x in x.split() if x in stop]))

# Number of numerics
df['numerics'] = df['Review'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

# lower-case
df['DETAIL'] = df['Review'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Removing Punctuation
df['DETAIL'] = df['DETAIL'].str.replace('[^\w\s]','')

#  Spelling correction
#from textblob import TextBlob
#df['DETAIL']= df['DETAIL'].apply(lambda x: str(TextBlob(x).correct()))

# remove stop words
from nltk.corpus import stopwords
stop = stopwords.words('english')
df['DETAIL'] = df['DETAIL'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# sentiment
df['sentiment'] = df['DETAIL'].apply(lambda x: TextBlob(x).sentiment[0] )
df['sentiment'] = np.round(df.sentiment, 1)

del df['Review']
df.rename(columns = {'Label':'rating'}, inplace = True)
# 1000 Common word removal
freq = pd.Series(' '.join(df['DETAIL']).split()).value_counts()[:1000]
freq = pd.DataFrame(freq)
freq.reset_index(inplace=True)
freq.columns = ['word', 'count']
#display(freq.shape[0])
display(freq.head())
#display(freq.tail())
tuples = [tuple(x) for x in freq.values]
b = {tuples[i]: tuples[i+1] for i in range(0, len(tuples), 2)}

from wordcloud import WordCloud
from PIL import Image
from matplotlib.pyplot import figure
import random

figure(num=None, figsize=(15, 12), dpi=80, facecolor='w', edgecolor='k')

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(206, 60%%, %d%%)" % random.randint(50, 70)

wc = WordCloud(background_color="#292929", max_words=1000, margin=0) 
wc.generate_from_frequencies(dict(tuples))

plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3), interpolation="bilinear")
plt.axis("off")
plt.show()

common_words = list(freq.word)
# 1000 rarw word removal
# 1000 Common word removal
freq = pd.Series(' '.join(df['DETAIL']).split()).value_counts()[-1000:]
freq = pd.DataFrame(freq)
freq.reset_index(inplace=True)
freq.columns = ['word', 'count']
#display(freq.shape[0])
display(freq.head())
#display(freq.tail())
tuples = [tuple(x) for x in freq.values]
b = {tuples[i]: tuples[i+1] for i in range(0, len(tuples), 2)}

from wordcloud import WordCloud
from PIL import Image
from matplotlib.pyplot import figure
import random

figure(num=None, figsize=(15, 12), dpi=80, facecolor='w', edgecolor='k')

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 60%%, %d%%)" % random.randint(50, 70)

wc = WordCloud(background_color="#292929", max_words=1000, margin=0) 
wc.generate_from_frequencies(dict(tuples))

plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3), interpolation="bilinear")
plt.axis("off")
plt.show()

rare_words = list(freq.word)
# Remove 1000 common and 1000 rare words 
# let’s remove these words as their presence will not of any use in classification of our text data
df['DETAIL'] = df['DETAIL'].apply(lambda x: " ".join(x for x in x.split() 
                                                     if x not in common_words))
df['DETAIL'] = df['DETAIL'].apply(lambda x: " ".join(x for x in x.split() 
                                                     if x not in rare_words))
df.head()
ratings = pd.DataFrame(df.groupby('rating').size()).reset_index()
ratings.columns = ['rating', 'count']
ratings
from matplotlib.pyplot import figure
#figure(num=None, figsize=(6, 8), dpi=80, facecolor='w', edgecolor='k')

sns.set(context='poster', style='ticks', font_scale=0.6)

sns.catplot(x="rating", y="count", kind='bar', palette="ch:.25", data=ratings)
plt.title('How are the rating?')
plt.xlabel('Rating')
plt.ylabel('Number of reviews')
sns.despine()
plt.tight_layout();
sentiment = pd.DataFrame(df.groupby(['sentiment'])['DETAIL'].count())
sentiment = sentiment.sort_values('sentiment', ascending=False)
sentiment.reset_index(inplace=True)

sentiment.columns = ['SENTIMENT', 'NUMBER_OF_REVIEWS']
#sentiment.tail()
from matplotlib.pyplot import figure
figure(num=None, figsize=(6, 8), dpi=80, facecolor='w', edgecolor='k')

sns.set(context='poster', style='ticks', font_scale=0.6)
# Reorder it following the values:
my_range=range(1,len(sentiment.index)+1)

# Create a color if the group is "B"
my_color=np.where(sentiment['SENTIMENT'] >= 0, '#5ab4ac', '#d8b365')
my_size=np.where(sentiment['SENTIMENT'] >= 0, 70, 30)

# The vertival plot is made using the hline function
# I load the seaborn library only to benefit the nice looking feature
plt.hlines(y=my_range, xmin=0, xmax=sentiment['NUMBER_OF_REVIEWS'], color=my_color, alpha=1)
plt.scatter(sentiment['NUMBER_OF_REVIEWS'], my_range, color=my_color, s=my_size, alpha=1)

# Add title and exis names

plt.yticks(my_range, sentiment['SENTIMENT'])
plt.title("What are the reviews saying?", loc='left')
plt.xlabel('Number of reviews')
plt.ylabel('Sentiment')
sns.despine();
