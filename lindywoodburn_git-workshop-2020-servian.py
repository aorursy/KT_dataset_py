print("Welcome to the GiT Workshop - Wisdom from Words")


# Standard python libraries

import re  # for regular expressions 

import numpy as np     

import pandas as pd

import random



# Libraries for visualisation

import matplotlib.pyplot as plt  

from wordcloud import WordCloud



# Gensim libraries - for building Word2Vec and Doc2Vec models

import gensim

from gensim.parsing.preprocessing import remove_stopwords

import nltk  # for text manipulation 



# XGBoost and Scikit-learn - for Classification model

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score



# Set log level

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)





# Workbook settings 

pd.set_option("display.max_colwidth", 200) 

%matplotlib inline





print("Libraries Loaded")
# Read the dataset

col_names = ["label", "ids", "date", "flag", "user", "tweet"]

df  = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv',encoding ="ISO-8859-1",names = col_names) 



# How many observations per sentiment?

print('Count of positive and negative tweets:')

print(df.label.value_counts())



# Look at the first five rows of data

df.head()
# Take a sample of the obervations

random.seed(18)

df = df.iloc[random.sample(range(df.shape[0]),100000)].copy()



# Keep ony the fields that we're interested in:

df = df[['ids','label','tweet']].copy()



# Update the Label to show 'Positive' and 'Negative'

df.label = df.label.replace({0: 'NEGATIVE', 4:'POSITIVE'})



# Check the number of observations in the sampled dataset:

print('Count of positive and negative tweets - sampled data:')

print(df.label.value_counts())
df[df['label'] == 'POSITIVE'].head(10)  
df[df['label'] == 'NEGATIVE'].head(10)  # Negative
# Start by copying all the tweet text to a new field - this is the field we'll manipulate

df['processed'] = df.tweet.str.lower()

df.processed = df.processed.str.replace('@[A-Za-z0-9_]+', '')   # removes all text starting with @ followed by an alpha character

df.processed = df.processed.str.replace('(www|http)\S+', '')    # removes urls

df.processed = df.processed.str.replace('&[a-z]+;', '')         # removes html expressions

df.processed = df.processed.str.replace('#[A-Za-z0-9_]+', '')   # removes all text starting with # followed by a word

df.processed = df.processed.str.replace('[^\sa-zA-Z0-9-]', ' ')   # keep only words and numbers ^==not

df.head(10)
# What stopwords will gensim remove?

all_stopwords = gensim.parsing.preprocessing.STOPWORDS

print([a for a in sorted(all_stopwords)])
# Remove the stop words

df['stopped'] = df.processed.apply(lambda x: remove_stopwords(str(x)))

df.head(10)
tokenizer = nltk.RegexpTokenizer(r"\w+")

df['tokens'] =df.stopped.apply(lambda x: [a for a in tokenizer.tokenize(str(x)) if len(a)>1])

df[['label','tweet','tokens']].head(10)
def p_color_func(word, font_size, position,orientation,random_state=None, **kwargs):

    return("hsl(270,100%%, %d%%)" % np.random.randint(10,80))



def b_color_func(word, font_size, position,orientation,random_state=None, **kwargs):

    return("hsl(200,100%%, %d%%)" % np.random.randint(10,80))



def make_word_cloud(word_list, color_func):

        wordcloud = WordCloud(width=1000, height=1000, 

                              background_color='white',

                              random_state=21, 

                              max_font_size=110,  

                              collocations= False).generate(' '.join(word_list))

        #change the color setting

        wordcloud.recolor(color_func = color_func)

        return wordcloud

    

neg_words = [word for words in df[df.label=='NEGATIVE'].tokens.values for word in words]

pos_words = [word for words in df[df.label=='POSITIVE'].tokens.values for word in words]



# Build Charts

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

#fig.suptitle('Horizontally stacked subplots')

axs[0].imshow(make_word_cloud(pos_words, p_color_func), interpolation="bilinear")

axs[0].set_title('POSITIVE Tweets')

axs[0].axis('off')

axs[1].imshow(make_word_cloud(neg_words, b_color_func), interpolation="bilinear")

axs[1].set_title('NEGATIVE Tweets')

axs[1].axis('off')

plt.tight_layout( w_pad=2, h_pad=0)

plt.show()
# Keep only those observations that have at least one 'token'

df= df[df.tokens.apply(len)>0]



# Split into train and test data sets (uses scikitlearn package for train/test split)

train_df, test_df = train_test_split(df, test_size=0.1, random_state=2020)



# Check the number of observations in the TRAIN dataset

print("TRAIN size:", len(train_df))

print('Count of positive and negative tweets - TRAIN data:')

print(train_df.label.value_counts())

# Check the number of observations in the TEST dataset

print("TEST size:", len(test_df))

print('Count of positive and negative tweets - TEST data:')

print(test_df.label.value_counts())
%%time



# Define the Gensim Model:

word2vec_model = gensim.models.word2vec.Word2Vec(size=300, 

                                        window=6, 

                                        min_count=10)



# Build the vocabulary

word2vec_model.build_vocab(train_df.tokens)
%%time



# Train the model to create the word vectors.

word2vec_model.train(train_df.tokens, total_examples=train_df.shape[0], epochs=5)
words = word2vec_model.wv.vocab.keys()

vocab_size = len(words)

vocab_counts = sorted([(w, word2vec_model.wv.vocab[w].count) for w in words], key = lambda x: x[1])

print("Vocab size", vocab_size)

# Look at the top 20 words

[w for w in vocab_counts][::-1][:20]
# What does the embedding look like?

word2vec_model.wv['girls'] 
word2vec_model.wv.most_similar('work')
word2vec_model.wv.most_similar('love')
word2vec_model.wv.most_similar('man')
# For the training and test dataset ... create a 'tagged' document. 



train_tagged = train_df.apply(

    lambda r: gensim.models.doc2vec.TaggedDocument(words=r.tokens, tags=[(r.label=='POSITIVE')*1]), axis=1)

test_tagged = test_df.apply(

    lambda r: gensim.models.doc2vec.TaggedDocument(words=r.tokens, tags=[(r.label=='POSITIVE')*1]), axis=1)

[x for x in train_tagged.values][:10]
%%time

# Build the distributed bag of words (like a vocab ... only multi-words) ~ 10 seconds

model_doc2vec = gensim.models.Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0)

model_doc2vec.build_vocab([x for x in train_tagged.values])
%%time

# Train the doc2vec model  --> This takes a little longer.  33 seconds

model_doc2vec.train(train_tagged.values, total_examples=len(train_tagged.values), epochs=3)

model_doc2vec.alpha = 0.002

model_doc2vec.min_alpha = model_doc2vec.alpha
print(train_df.tweet[train_df.index==1158434])

print(train_tagged[1158434])
# Helper function to generate the document vectors for each vector

def vec_for_learning(model, tagged_docs):

    sents = tagged_docs.values

    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])

    return targets, regressors



vec_for_learning(model_doc2vec, train_tagged[train_tagged.index==1158434])
%%time

# Prepare the data for training ... converts the word into the embeddings discovered by our Doc2Vec model  ~ 3 minutes

# 5,000 =  1 min 71 - 4 minutes%, 10,000 = 1min 39s 72%; 50,000 6min 52, 100,000 13 mins 72% .... variable on Kaggle!

y_train, X_train = vec_for_learning(model_doc2vec, train_tagged)

y_test, X_test = vec_for_learning(model_doc2vec, test_tagged)





xgb_model = XGBClassifier()

xgb_model.fit(pd.DataFrame(X_train).head(5000), y_train[:5000]) 
# Predict using the test set to determine accuracy

y_pred = xgb_model.predict(pd.DataFrame(X_test))



print('Testing accuracy %s' % accuracy_score(y_test, y_pred))

print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
tv = model_doc2vec.infer_vector('congratulations you made it this far'.split(),steps = 20)

tv
xgb_model.predict(pd.DataFrame([tv]))[0]
def classify_my_tweet():

        tweet_text = input("\n\tType in some text: ")

        tweet_text = tweet_text.lower()

        

        tweet_vector = model_doc2vec.infer_vector(tweet_text.split(),steps = 50)

        answer = xgb_model.predict(pd.DataFrame([tweet_vector]))

        print ('\n\t\tPOSITIVE\n\n\t\t( ͡♥ ͜ʖ ͡♥)\n' if answer[0]==1 

               else '\n\t\tNEGATIVE\n\n\t\t( ˘︹˘ )\n')

# Prompt for input and classify the sentiment

#classify_my_tweet()

def classify_tweet_list(tweets):

        for twt in tweets:

            tweet_vector = model_doc2vec.infer_vector(twt.lower().split(),steps = 50)

            answer = xgb_model.predict(pd.DataFrame([tweet_vector]))

            print ('\n  POSITIVE\t( ͡♥ ͜ʖ ͡♥)' if answer[0]==1 

                   else '\n  NEGATIVE\t( ˘︹˘ )', '\t',twt)





pretend_tweets = ['why does it always rain',

                  'winning the race today',

                  'I like chocolate and coffee',

                  'feeling sleepy after too much food',

                  'yay its pay day',

                  'the sky is blue',

                  'the sky is grey',

                  'feeling kinda blue',

                  'on top of the world',

                  'feeling happy about today'

                                     ]

classify_tweet_list(pretend_tweets)

 