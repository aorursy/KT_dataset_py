# Packages

import os

import numpy as np

import pandas as pd

import nltk

import random



# Pre-Processing

import string

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

import re

from nltk.stem import PorterStemmer

from nltk.stem.lancaster import LancasterStemmer

from nltk.stem.porter import *



# Sentiment Analysis

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.sentiment.util import *

import matplotlib.pyplot as plt



# Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS

import seaborn as sns



# N- Grams

from nltk.util import ngrams

from collections import Counter



# Topic Modeling

from nltk import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation



# Word 2 Vec

from gensim.models import Word2Vec

from sklearn.decomposition import PCA



# Models

import datetime

from nltk import naivebayes



import warnings

warnings.filterwarnings("ignore")
import sys

sys.version
# Read Data

df = pd.read_csv("../input/train.csv", index_col="id")

test = pd.read_csv("../input/test.csv", index_col="id")
pd.set_option('max_colwidth', 500)

df.text= df.text.astype(str)

df.author = pd.Categorical(df.author)

df.iloc[:20,:]
from nltk.stem.lancaster import LancasterStemmer

from nltk.stem.porter import *

#ps = LancasterStemmer()

ps = PorterStemmer()



tokenizer = RegexpTokenizer(r'\w+')

stop_words = set(stopwords.words('english'))



def preprocessing(data):

    txt = data.str.lower().str.cat(sep=' ') #1

    words = tokenizer.tokenize(txt) #2

    words = [w for w in words if not w in stop_words] #3

    #ords = [ps.stem(w) for w in words] #4

    return words



def wordfreqviz(text, x):

    word_dist = nltk.FreqDist(text)

    top_N = x

    rslt = pd.DataFrame(word_dist.most_common(top_N),

                    columns=['Word', 'Frequency']).set_index('Word')

    matplotlib.style.use('ggplot')

    rslt.plot.bar(rot=0)



def wordfreq(text, x):

    word_dist = nltk.FreqDist(text)

    top_N = x

    rslt = pd.DataFrame(word_dist.most_common(top_N),

                    columns=['Word', 'Frequency']).set_index('Word')

    print(rslt)
# Pre-Processing

SIA = SentimentIntensityAnalyzer()



# Applying Model, Variable Creation

sentiment = df.copy()

sentiment['polarity_score']=sentiment.text.apply(lambda x:SIA.polarity_scores(x)['compound'])

sentiment['neutral_score']=sentiment.text.apply(lambda x:SIA.polarity_scores(x)['neu'])

sentiment['negative_score']=sentiment.text.apply(lambda x:SIA.polarity_scores(x)['neg'])

sentiment['positive_score']=sentiment.text.apply(lambda x:SIA.polarity_scores(x)['pos'])

sentiment['sentiment']=''

sentiment.loc[sentiment.polarity_score>0,'sentiment']='POSITIVE'

sentiment.loc[sentiment.polarity_score==0,'sentiment']='NEUTRAL'

sentiment.loc[sentiment.polarity_score<0,'sentiment']='NEGATIVE'



# Normalize for Size

auth_sent= sentiment.groupby(['author','sentiment'])[['text']].count().reset_index()

for x in ['EAP','HPL','MWS']:

    auth_sent.text[auth_sent.author == x] = (auth_sent.text[auth_sent.author == x]/\

        auth_sent[auth_sent.author ==x].text.sum())*100
ax= sns.barplot(x='sentiment', y='text',hue='author',data=auth_sent)

ax.set(xlabel='Author', ylabel='Sentiment Percentage')

ax.figure.suptitle("Author by Sentiment", fontsize = 24)

plt.show()
# Function

def cloud(text, title):

    # Setting figure parameters

    mpl.rcParams['figure.figsize']=(10.0,10.0)    #(6.0,4.0)

    #mpl.rcParams['font.size']=12                #10 

    mpl.rcParams['savefig.dpi']=100             #72 

    mpl.rcParams['figure.subplot.bottom']=.1 

    

    # Processing Text

    stopwords = set(STOPWORDS) # Redundant

    wordcloud = WordCloud(width=1400, height=800,

                          background_color='black',

                          #stopwords=stopwords,

                         ).generate(" ".join(text))

    

    # Output Visualization

    plt.figure(figsize=(20,10), facecolor='k')

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.tight_layout(pad=0)

    plt.title(title, fontsize=50,color='y')

    #plt.imshow(plt.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)

    #fig.savefig("wordcloud.png", dpi=900)
x = "EAP"

print(cloud(df[df.author == x]['text'].values,x))
x = "HPL"

print(cloud(df[df.author == x]['text'].values,x))
x = "MWS"

print(cloud(df[df.author == x]['text'].values,x))
## Helper Functions

def get_ngrams(text, n):

    n_grams = ngrams((text), n)

    return [ ' '.join(grams) for grams in n_grams]



def gramfreq(text,n,num):

    # Extracting bigrams

    result = get_ngrams(text,n)

    # Counting bigrams

    result_count = Counter(result)

    # Converting to the result to a data frame

    df = pd.DataFrame.from_dict(result_count, orient='index')

    df = df.rename(columns={'index':'words', 0:'frequency'}) # Renaming index column name

    return df.sort_values(["frequency"],ascending=[0])[:num]



def gram_table(x, gram, length):

    out = pd.DataFrame(index=None)

    for i in gram:

        table = pd.DataFrame(gramfreq(preprocessing(df[df.author == x]['text']),i,length).reset_index())

        table.columns = ["{}-Gram".format(i),"Occurence"]

        out = pd.concat([out, table], axis=1)

    return out
gram_table(x="EAP", gram=[1,2,3,4], length=20)
gram_table(x="HPL", gram=[1,2,3,4], length=20)
gram_table(x="MWS", gram=[1,2,3,4], length=20)
lemm = WordNetLemmatizer()

class LemmaCountVectorizer(CountVectorizer):

    def build_analyzer(self):

        analyzer = super(LemmaCountVectorizer, self).build_analyzer()

        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))

    

# Define helper function to print top words

def print_top_words(model, feature_names, n_top_words):

    for index, topic in enumerate(model.components_):

        message = "\nTopic #{}:".format(index)

        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])

        print(message)

        print("="*70)

    

def LDA(data):

    # Storing the entire training text in a list

    text = list(data.values)

    # Calling our overwritten Count vectorizer

    tf_vectorizer = LemmaCountVectorizer(max_df=0.95, min_df=2,

                                              stop_words='english',

                                              decode_error='ignore')

    tf = tf_vectorizer.fit_transform(text)





    lda = LatentDirichletAllocation(n_topics=6, max_iter=5,

                                    learning_method = 'online',

                                    learning_offset = 50.,

                                    random_state = 0)



    lda.fit(tf)



    n_top_words = 10

    print("\nTopics in LDA model: ")

    tf_feature_names = tf_vectorizer.get_feature_names()

    print_top_words(lda, tf_feature_names, n_top_words)
x = "EAP"

LDA(df.text[df.author==x])
x = "MWS"

LDA(df.text[df.author==x])
x = "HPL"

LDA(df.text[df.author==x])
def model_prep(df_in):

    df_in['tokenized'] = df_in.text.astype(str).str.lower() # turn into lower case text

    df_in['tokenized'] = df_in.apply(lambda row: tokenizer.tokenize(row['tokenized']), axis=1) # apply tokenize to each row

    df_in['tokenized'] = df_in['tokenized'].apply(lambda x: [w for w in x if not w in stop_words]) # remove stopwords from each row

    #df_in['tokenized'] = df_in['tokenized'].apply(lambda x: [ps.stem(w) for w in x]) # apply stemming to each row

    return df_in



def w2vec(data,yrange):

    wvec = model_prep(data)

    model = Word2Vec(data.tokenized, min_count=1, max_vocab_size=250)

    # model.save('model.bin')

    # new_model = Word2Vec.load('model.bin')

    

    # summarize the loaded model

    print(model)



    X = model[model.wv.vocab]

    pca = PCA(n_components=2)

    result = pca.fit_transform(X)

    # create a scatter plot of the projection

    plt.rcParams["figure.figsize"] = [16,9]



    plt.scatter(result[:, 0], result[:, 1])

    words = list(model.wv.vocab)

    for i, word in enumerate(words):

        plt.annotate(word, xy=(result[i, 0], result[i, 1]))



    plt.ylim(yrange)   



    plt.show()

    

x = "MWS"

w2vec(df[df.author==x],[-.015,.015])
x = "EAP"

print("\n",x)

w2vec(df[df.author==x],[-.014,.014])
x = "HPL"

print("\n",x)

w2vec(df[df.author==x],[-.015,.015])
print("Train Vocabulary Size: {}".format(len(nltk.FreqDist(preprocessing(df['text'])))))

print("Train Size: {}".format(len(df)))

print("Test Vocabulary Size: {}".format(len(nltk.FreqDist(preprocessing(test['text'])))))

print("Test Size: {}".format(len(test)))
# Number of features

all_words = nltk.FreqDist(preprocessing(df['text'])) # Calculate word occurence from whole block of text

word_features = list(all_words.keys())[:20000] 

# Number of columns (can't exceed vocab, only shrink it) from largest to smallest



# Helper Functions

# for each review, records which uniqeue words out of the whole text body are present

def find_features(document):

    words = set(document)

    features = {}

    for w in word_features:

        features[w] = (w in words)



    return features



# Function to create model features

def model_prep(state, df_in):

    df_in['tokenized'] = df_in.text.astype(str).str.lower() # turn into lower case text

    df_in['tokenized'] = df_in.apply(lambda row: tokenizer.tokenize(row['tokenized']), axis=1) # apply tokenize to each row

    df_in['tokenized'] = df_in['tokenized'].apply(lambda x: [w for w in x if not w in stop_words]) # remove stopwords from each row

    df_in['tokenized'] = df_in['tokenized'].apply(lambda x: [ps.stem(w) for w in x]) # apply stemming to each row

    if state == "Train":

        print("{} Word Features: {}".format(state, len(word_features)))

        print("All Possible words in {} set: {}".format(state, len(all_words)))

        # Bag of Words with Label

        featuresets = [(find_features(text), LABEL) for (text, LABEL) in list(zip(df_in.tokenized, (df_in.author)))]

        print("Train Set Size: {}".format(len(featuresets)))

        print("Train Set Ready")

        return featuresets, word_features

    else:

        # Bag of Words without Labels

        featuresets = [(find_features(text)) for (text) in list(df_in.tokenized)]

        print("Submission Set Size: {}".format(len(featuresets)))

        print("Submission Set Ready")

        return featuresets
trainset, word_features= model_prep("Train", df_in=df)
submissionset = model_prep("Test", df_in=test)
training_set = trainset[:15000]

testing_set = trainset[15000:]

del trainset
start = time.time()

classifier = nltk.NaiveBayesClassifier.train(training_set)

# Posterior = prior_occurence * likelihood / evidence

end = time.time()

print("Model took %0.2f seconds to train"%(end - start))
# Edgar Allan Poe [EAP], Mary Shelley[MWS], and HP Lovecraft[HPL]
print("Classifier Test Accuracy:",(nltk.classify.accuracy(classifier, testing_set))*100)

print(classifier.show_most_informative_features(40))
classifier.labels()
labels = classifier.labels()

submission = pd.DataFrame(columns=labels)

for x in submissionset:

    dist = classifier.prob_classify(x)

    submission= submission.append({labels[0]:dist.prob(labels[0]),

                                   labels[1]:dist.prob(labels[1]),

                                   labels[2]:dist.prob(labels[2])},ignore_index=True)

submission["id"] = test.index

submission= submission[["id", "EAP","HPL","MWS"]]
submission.head()
submission.to_csv("naive_spooky.csv", index=False)