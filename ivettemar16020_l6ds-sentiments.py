import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



import nltk

from nltk.corpus import stopwords

from nltk.classify import SklearnClassifier

from nltk import sent_tokenize, word_tokenize, pos_tag

from nltk.tokenize import RegexpTokenizer

from nltk.stem import WordNetLemmatizer, PorterStemmer

from nltk import FreqDist



from wordcloud import WordCloud, STOPWORDS



import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter

%matplotlib inline



from textblob import TextBlob
pd_data = pd.read_csv("../input/grammar-and-online-product-reviews/GrammarandProductReviews.csv")

pd_data = pd_data[['brand','categories', 'name', 'reviews.didPurchase', 'reviews.doRecommend', 'reviews.numHelpful', 'reviews.rating', 'reviews.text', 'reviews.title']]
def remove_punctuation(text): 

    no_punct = "".join([c for c in text if c not in string.punctuation])

    return no_punct
#TOKENIZER, LOWERCASE AND REMOVE PUNCTUATION

tokenizer = RegexpTokenizer(r'\w+')



pd_data['brand'] = pd_data['brand'].apply(lambda x: tokenizer.tokenize(x.lower()))

pd_data['categories'] = pd_data['categories'].apply(lambda x: tokenizer.tokenize(x.lower()))

pd_data['name'] = pd_data['name'].apply(lambda x: tokenizer.tokenize(x.lower()))

pd_data['reviews.text'] = pd_data['reviews.text'].apply(lambda x: tokenizer.tokenize(str(x).lower()))

pd_data['reviews.title'] = pd_data['reviews.title'].apply(lambda x: tokenizer.tokenize(str(x).lower()))
pd_data['reviews.text'].head(20)
def remove_stopwords(text): 

    words = [w for w in text if w not in stopwords.words('english')]

    return words
#pd_data['brand'] = pd_data['brand'].apply(lambda x: remove_stopwords(x))

#pd_data['categories'] = pd_data['categories'].apply(lambda x: remove_stopwords(x))

#pd_data['name'] = pd_data['name'].apply(lambda x: remove_stopwords(x))

pd_data['reviews.text'] = pd_data['reviews.text'].apply(lambda x: remove_stopwords(x))

pd_data['reviews.title'] = pd_data['reviews.title'].apply(lambda x: remove_stopwords(x))
pd_data['reviews.text'].head(20)
#Lemmatizer

lemmatizer = WordNetLemmatizer()



def word_lemmatizer(text): 

    lem_text = [lemmatizer.lemmatize(i) for i in text]

    return lem_text
pd_data['reviews.text'].apply(lambda x: word_lemmatizer(x))
#Stemmer

stemmer = PorterStemmer()



def word_stemmer(text): 

    stem_text = " ".join([stemmer.stem(i) for i in text])

    return stem_text
pd_data['reviews.text'].apply(lambda x: word_stemmer(x))
pd_data['reviews.text'] = pd_data['reviews.text'].apply(lambda x: word_stemmer(x))
dfText = pd_data['reviews.text'].str.split(expand=True).stack().value_counts()
dfText[:20].plot.bar()
dfText[-20:].plot.bar()
#WORD CLOUD

all_text = pd_data['reviews.text'].str.split(' ')



text_data = [" ".join(text) for text in all_text]

final_text = " ".join(text_data)



wordcloud_text = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(final_text)
#PLOT WORD CLOUD

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud_text, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
def get_sentiment(text): 

    analysis = TextBlob(text) 

    # set sentiment 

    if analysis.sentiment.polarity > 0: 

        return 'positive'

    elif analysis.sentiment.polarity == 0: 

        return 'neutral'

    else: 

        return 'negative'
clasify_sent = pd_data['reviews.text'].apply(lambda x: get_sentiment(x))
number = clasify_sent.value_counts()

number

number.plot.bar()
pd_data['sentiment'] = clasify_sent

pd_data[['reviews.text', 'sentiment']]