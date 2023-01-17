import warnings   # To avoid warning messages in the code run

warnings.filterwarnings("ignore")



%matplotlib inline  

# To make data visualisations display in Jupyter Notebooks 

import numpy as np   # linear algebra

import pandas as pd  # Data processing, Input & Output load



import matplotlib.pyplot as plt # Visuvalization & plotting

import seaborn as sns  #Data visualisation



import nltk # Natural Language Toolkit (statistical natural language processing (NLP) libraries )

from nltk.stem.porter import *   # Stemming 



from sklearn.model_selection import train_test_split, cross_val_score

                                    # train_test_split - Split arrays or matrices into random train and test subsets

                                    # cross_val_score - Evaluate a score by cross-validation



from sklearn.ensemble import RandomForestClassifier # RandomForestClassifier model to predict sentiment



from sklearn.feature_extraction.text import CountVectorizer #CountVectorizer converts collection of text docs to a matrix of token counts



from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer # Converting occurrences to frequencies



from sklearn.metrics import classification_report
Tweets = pd.read_csv("../input/Sentiment.csv",sep=",") 
print(Tweets.shape)

Tweets.head()
Tweet_Sentiment = Tweets[['sentiment','text']]

print(Tweet_Sentiment.shape)

Tweet_Sentiment.head()
set(Tweet_Sentiment.sentiment)
print("========*** Sentiment counts ***=============")

dist = Tweet_Sentiment.groupby(["sentiment"]).size()

print(dist)



print("========*** Sentiment Percentage ***=============")



dist_Percentage = round((dist / dist.sum())*100,2)

print(dist_Percentage)
Tweet_Sentiment['sentiment'].value_counts().plot(kind = 'bar')
Tweet_Sentiment['word_count'] = Tweet_Sentiment['text'].apply(lambda x: len(str(x).split(" ")))

Tweet_Sentiment[['text','word_count']].head()
Tweet_Sentiment['PP_text'] = Tweet_Sentiment['text'].str.replace("@[^\s]+", "at_user")

Tweet_Sentiment['PP_text'].head()
Tweet_Sentiment['PP_text'] = Tweet_Sentiment['PP_text'].str.replace('[^\w\s]','')
Tweet_Sentiment['PP_text'] = Tweet_Sentiment['PP_text'].str.replace('((www\.[^\s]+)|(https?://[^\s]+))','')
Tweet_Sentiment['PP_text'] = Tweet_Sentiment['PP_text'].str.lower()

Tweet_Sentiment[['text','PP_text']].head()
from nltk.corpus import stopwords

stop = stopwords.words('english')



Tweet_Sentiment['PP_text'] = Tweet_Sentiment['PP_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

Tweet_Sentiment['PP_text'].head()
freq = pd.Series(' '.join(Tweet_Sentiment['PP_text']).split()).value_counts()[:10]

freq
freq = list(freq.index)

Tweet_Sentiment['PP_text'] = Tweet_Sentiment['PP_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

Tweet_Sentiment['PP_text'].head()
freq = pd.Series(' '.join(Tweet_Sentiment['PP_text']).split()).value_counts()[-50:]

print(freq)

freq = list(freq.index)

Tweet_Sentiment['PP_text'] = Tweet_Sentiment['PP_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

Tweet_Sentiment['PP_text'].head()
Tweet_Sentiment['tokenized_words'] = Tweet_Sentiment['PP_text'].apply(lambda x: x.split())

Tweet_Sentiment.tokenized_words.head()
from textblob import Word

Tweet_Sentiment['PP_text'] = Tweet_Sentiment['PP_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

Tweet_Sentiment['PP_text'].head()
Positive_tweets = Tweet_Sentiment[ Tweet_Sentiment['sentiment'] == 'Positive']

Positive_tweets = Positive_tweets['PP_text']

print(Positive_tweets)



Negative_tweets = Tweet_Sentiment[ Tweet_Sentiment['sentiment'] == 'Negative']

Negative_tweets = Negative_tweets['PP_text']

print(Negative_tweets)

from wordcloud import WordCloud
comb = " ".join(Positive_tweets)



wordcloud = WordCloud().generate(comb)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off");
comb = " ".join(Negative_tweets)



wordcloud = WordCloud().generate(comb)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off");
TF = (Tweet_Sentiment['PP_text'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()

TF.columns = ['words','tf']

#tf.sort('tf')

TF

#Tweet_Sentiment['PP_text'][1:2]
for i,word in enumerate(TF['words']):

  TF.loc[i, 'idf'] = np.log(Tweet_Sentiment.shape[0]/(len(Tweet_Sentiment[Tweet_Sentiment['PP_text'].str.contains(word)])))



TF
TF['tfidf'] = TF['tf'] * TF['idf']

TF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,1))

train_vect = tfidf.fit_transform(Tweet_Sentiment['PP_text'])



print(tfidf)

train_vect.shape

tfidf.vocabulary_