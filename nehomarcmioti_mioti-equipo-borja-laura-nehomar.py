#Básicos

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import re

import itertools

import datetime



from textblob import TextBlob



# NLTK

import nltk

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer 

from nltk.tokenize import word_tokenize



import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

import unidecode

import string



from nltk.probability import FreqDist
train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

train_df = train_df[train_df['text'].notna()]

train_df = train_df.reset_index()

train_df.head(10)
train_df.info()
stop_words = set(stopwords.words('english'))



appos = {

"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"don't" : "do not",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"he'd" : "he would",

"he'll" : "he will",

"he's" : "he is",

"i'd" : "i would",

"i'd" : "i had",

"i'll" : "i will",

"i'm" : "i am",

"isn't" : "is not",

"it's" : "it is",

"it'll":"it will",

"i've" : "i have",

"let's" : "let us",

"mightn't" : "might not",

"mustn't" : "must not",

"shan't" : "shall not",

"she'd" : "she would",

"she'll" : "she will",

"she's" : "she is",

"shouldn't" : "should not",

"that's" : "that is",

"there's" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"they've" : "they have",

"we'd" : "we would",

"we're" : "we are",

"weren't" : "were not",

"we've" : "we have",

"what'll" : "what will",

"what're" : "what are",

"what's" : "what is",

"what've" : "what have",

"where's" : "where is",

"who'd" : "who would",

"who'll" : "who will",

"who're" : "who are",

"who's" : "who is",

"who've" : "who have",

"won't" : "will not",

"wouldn't" : "would not",

"you'd" : "you would",

"you'll" : "you will",

"you're" : "you are",

"you've" : "you have",

"'re": " are",

"wasn't": "was not",

"we'll":" will",

"didn't": "did not"

}
def text_preprocess(text):

    lemma = nltk.wordnet.WordNetLemmatizer()

    

    text = str(text)

    

    #removing mentions and hashtags



    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", text).split())

    

    #remove http links from tweets

    

    

    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)

    links         = re.findall(link_regex, text)

    for link in links:

        text = text.replace(link[0], '')  

    

    text_pattern = re.sub("`", "'", text)

    

    #fix misspelled words

    #Para esto solo se comprueba que no hayan palabras con dos letras seguidas iguales.'



    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))

    

    

   # print(text_pattern)

    

    #Convert to lower and negation handling

    

    text_lr = text_pattern.lower()

    

   # print(text_lr)

    

    words = text_lr.split()

    text_neg = [appos[word] if word in appos else word for word in words]

    text_neg = " ".join(text_neg) 

   # print(text_neg)

    

    #remove stopwords

    tokens = word_tokenize(text_neg)

    text_nsw = [i for i in tokens if i not in stop_words]

    text_nsw = " ".join(text_nsw) 

   # print(text_nsw)

    

    

    #remove tags

    text_tags=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text_nsw)



    # remove special characters and digits

    text_alpha=re.sub("(\\d|\\W)+"," ",text_tags)

    

    #Remove accented characters

    text = unidecode.unidecode(text_alpha)

    

    '''#Remove punctuation

    table = str.maketrans('', '', string.punctuation)

    text = [w.translate(table) for w in text.split()]'''

    

    sent = TextBlob(text)

    tag_dict = {"J": 'a', 

                "N": 'n', 

                "V": 'v', 

                "R": 'r'}

    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    

    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]

   

    return " ".join(lemmatized_list)
train_df['processed_text'] = None



for i in range(len(train_df)):

    train_df.processed_text[i] = text_preprocess(train_df.text[i])
train_df.head(10)
train_df.tail(10)
import matplotlib.pyplot as plt

ax = train_df['sentiment'].value_counts(sort=False).plot(kind='barh')

ax.set_xlabel('Número de muestras')

ax.set_ylabel('Etiqueta')
from wordcloud import WordCloud

import matplotlib.pyplot as plt



# Polarity ==  negative

train_s0 = train_df[train_df.sentiment == 'negative']

all_text = ' '.join(word for word in train_s0.processed_text)

wordcloud_neg = WordCloud(colormap='Reds', width=1000, height=1000, background_color='white').generate(all_text) #mode='RGBA'

plt.figure(figsize=(20,10))

plt.title('Negative sentiment - Wordcloud')

plt.imshow(wordcloud_neg, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()



wordcloud_neg.to_file('negative_senti_wordcloud.jpg')



# Polarity ==  neutral

train_s1 = train_df[train_df.sentiment == 'neutral']

all_text = ' '.join(word for word in train_s1.processed_text)

wordcloud_neu = WordCloud(width=1000, height=1000, colormap='Blues', background_color='white').generate(all_text)

plt.figure( figsize=(20,10))

plt.title('Neutral sentiment - Wordcloud')

plt.imshow(wordcloud_neu, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()



wordcloud_neu.to_file('neutral_senti_wordcloud.jpg')



# Polarity ==  positive

train_s2 = train_df[train_df.sentiment  == 'positive']

all_text = ' '.join(word for word in train_s2.processed_text)

wordcloud_pos = WordCloud(width=1000, height=1000, colormap='Wistia',background_color='white').generate(all_text)

plt.figure(figsize=(20,10))

plt.title('Positive sentiment - Wordcloud')

plt.imshow(wordcloud_pos, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()



wordcloud_pos.to_file('positive_senti_wordcloud.jpg')