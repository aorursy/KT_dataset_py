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

from sklearn.preprocessing import LabelEncoder # Labeling the columns with 0 & 1

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import sent_tokenize, word_tokenize
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
comb = " ".join(Positive_tweets)



wordcloud = WordCloud().generate(comb)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off");
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#Tweet_Sentiment['PP_text']

Sentiment_df = Tweet_Sentiment.copy()

sentiment = SentimentIntensityAnalyzer()



i=0

for sentence in Tweet_Sentiment['PP_text']:

    vs = sentiment.polarity_scores(sentence)

    Sentiment_df.loc[i, 'Sentiment_Pred'] = vs.get('compound')

    i+=1
Sentiment_df.head()
# Functions  convert the scores into categories. The compound scores extracted is a value between -1 & +1. 

# Values towards neagtive means negative sentiment and vice versa



def cat(x):

    if -1<=x<0:

        return "Negative"

    if x==0:

        return "Neutral"

    if 0<x<=1:

        return "Positive"



Sentiment_df["Sentiment_Cat"] = Sentiment_df[["Sentiment_Pred"]].applymap(cat)

Sentiment_df.head()
tab = pd.crosstab(Sentiment_df['sentiment'], Sentiment_df['Sentiment_Cat'])

tab
print(classification_report(Sentiment_df['sentiment'], Sentiment_df['Sentiment_Cat']))
print('Accuracy - ', np.round((np.trace(tab)/Sentiment_df.shape[0])*100,2), "%")
def cat_num(x):

    if -1<=x<-0.5:

        return 0

    if -0.5<=x<0:

        return 1

    if x==0:

        return 2

    if 0<x<=0.5:

        return 3

    return 4
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
# Get all the stop words and punctuations

import string

stop_words = set(stopwords.words('english'))

punct = string.punctuation
# Remove all stop_words and punctuations before tokenizing so that we have only meaningful words



stop_words_union = stop_words.union(punct)

stop_words_union
lemma = WordNetLemmatizer()



def create_lemmas(sentence):

    sentence=sentence.lower()

    words = word_tokenize(sentence)

    words_cleaned=[]

    for word in words :

        if word in stop_words_union:continue

        words_cleaned.append(word)

    return [lemma.lemmatize(word, pos="v") for word in words_cleaned]
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(analyzer=create_lemmas,max_features=1000, lowercase=True,stop_words=stop_words_union,ngram_range=(1,1))

train_vect = tfidf.fit_transform(Tweet_Sentiment['PP_text'])



print(tfidf)

train_vect.shape

tfidf.vocabulary_
le=LabelEncoder()

y=le.fit_transform(Tweet_Sentiment.sentiment.values)

#le.fit_transform(Tdata[i])

#Tweet_Sentiment.sentiment.values\

print(set(y))
X_train_vect, X_test_vect, y_train_vect, y_test_vect = train_test_split(train_vect, y, train_size=0.75)



print(X_train_vect.shape)

print(X_test_vect.shape)

print(y_train_vect.shape)

print(y_test_vect.shape)
Model = MultinomialNB()  ## Multinomial Naive Bayes
Model.fit(X_train_vect, y_train_vect)

predictions_nb = Model.predict(X_test_vect)

accuracy = accuracy_score(y_test_vect, predictions_nb)
print("MultinomialNB Accuracy:",accuracy_score(y_test_vect, predictions_nb))
RF_model = RandomForestClassifier(class_weight='balanced')

Final_Model = RF_model.fit(X_train_vect, y_train_vect)
predictions_RF = RF_model.predict(X_test_vect)

accuracy = accuracy_score(y_test_vect, predictions_RF)

print("RandomForestmodel Accuracy:",accuracy_score(y_test_vect, predictions_RF))