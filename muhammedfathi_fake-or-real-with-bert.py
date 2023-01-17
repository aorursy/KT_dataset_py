import pandas as pd

import numpy as np

import os



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns



from nltk.corpus import stopwords

from nltk.util import ngrams



from wordcloud import WordCloud



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.metrics import classification_report,confusion_matrix



from collections import defaultdict

from collections import Counter

plt.style.use('ggplot')

stop=set(stopwords.words('english'))



import re

from nltk.tokenize import word_tokenize

import gensim

import string



from tqdm import tqdm

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding, LSTM,Dense, SpatialDropout1D, Dropout

from keras.initializers import Constant

from keras.optimizers import Adam
tweet= pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')

submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
# # From https://www.kaggle.com/wrrosa/keras-bert-using-tfhub-modified-train-data - 

# # author of this kernel read tweets in training data and figure out that some of them have errors:

# ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]

# tweet.loc[tweet['id'].isin(ids_with_target_error),'target'] = 0

# tweet[tweet['id'].isin(ids_with_target_error)]
print('There are {} rows and {} columns in train'.format(tweet.shape[0],tweet.shape[1]))

print('There are {} rows and {} columns in train'.format(test.shape[0],test.shape[1]))
tweet.head(10)
# extracting the number of examples of each class

Real_len = tweet[tweet['target'] == 1].shape[0]

Not_len = tweet[tweet['target'] == 0].shape[0]
# bar plot of the 3 classes

plt.rcParams['figure.figsize'] = (7, 5)

plt.bar(10,Real_len,3, label="Real", color='blue')

plt.bar(15,Not_len,3, label="Not", color='red')

plt.legend()

plt.ylabel('Number of examples')

plt.title('Propertion of examples')

plt.show()
def length(text):    

    '''a function which returns the length of text'''

    return len(text)
tweet['length'] = tweet['text'].apply(length)
plt.rcParams['figure.figsize'] = (18.0, 6.0)

bins = 150

plt.hist(tweet[tweet['target'] == 0]['length'], alpha = 0.6, bins=bins, label='Not')

plt.hist(tweet[tweet['target'] == 1]['length'], alpha = 0.8, bins=bins, label='Real')

plt.xlabel('length')

plt.ylabel('numbers')

plt.legend(loc='upper right')

plt.xlim(0,150)

plt.grid()

plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_len=tweet[tweet['target']==1]['text'].str.len()

ax1.hist(tweet_len,color='blue')

ax1.set_title('disaster tweets')

tweet_len=tweet[tweet['target']==0]['text'].str.len()

ax2.hist(tweet_len,color='red')

ax2.set_title('Not disaster tweets')

fig.suptitle('Characters in tweets')

plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_len=tweet[tweet['target']==1]['text'].str.split().map(lambda x: len(x))

ax1.hist(tweet_len,color='blue')

ax1.set_title('disaster tweets')

tweet_len=tweet[tweet['target']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_len,color='red')

ax2.set_title('Not disaster tweets')

fig.suptitle('Words in a tweet')

plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

word=tweet[tweet['target']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='blue')

ax1.set_title('disaster')

word=tweet[tweet['target']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='red')

ax2.set_title('Not disaster')

fig.suptitle('Average word length in each tweet')
def create_corpus(target):

    corpus=[]

    

    for x in tweet[tweet['target']==target]['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus
def create_corpus_df(tweet, target):

    corpus=[]

    

    for x in tweet[tweet['target']==target]['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus
corpus=create_corpus(0)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1

        

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]
# displaying the stopwords

np.array(stop)
plt.rcParams['figure.figsize'] = (18.0, 6.0)

x,y=zip(*top)

plt.bar(x,y)
corpus=create_corpus(1)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1



top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

    



plt.rcParams['figure.figsize'] = (18.0, 6.0)

x,y=zip(*top)

plt.bar(x,y)
plt.figure(figsize=(16,5))

corpus=create_corpus(1)



dic=defaultdict(int)

special = string.punctuation

for i in (corpus):

    if i in special:

        dic[i]+=1

        

x,y=zip(*dic.items())

plt.bar(x,y)
plt.figure(figsize=(16,5))

corpus=create_corpus(0)

dic=defaultdict(int)

special = string.punctuation

for i in (corpus):

    if i in special:

        dic[i]+=1

        

x,y=zip(*dic.items())

plt.bar(x,y,color='green')
plt.figure(figsize=(16,5))

counter=Counter(corpus)

most=counter.most_common()

x=[]

y=[]

for word,count in most[:40]:

    if (word not in stop) :

        x.append(word)

        y.append(count)
sns.barplot(x=y,y=x)
def get_top_tweet_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
plt.figure(figsize=(16,5))

top_tweet_bigrams=get_top_tweet_bigrams(tweet['text'])[:10]

x,y=map(list,zip(*top_tweet_bigrams))

sns.barplot(x=y,y=x)
df=pd.concat([tweet,test])

df.shape
example="New competition launched :https://www.kaggle.com/c/nlp-getting-started"
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



remove_URL(example)
df['text']=df['text'].apply(lambda x : remove_URL(x))
example = """<div>

<h1>Real or Fake</h1>

<p>Kaggle </p>

<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>

</div>"""
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

print(remove_html(example))
df['text']=df['text'].apply(lambda x : remove_html(x))
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



remove_emoji("Omg another Earthquake 😔😔")
df['text']=df['text'].apply(lambda x: remove_emoji(x))
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



example="I am a #king"

print(remove_punct(example))
df['text']=df['text'].apply(lambda x : remove_punct(x))
corpus_new1=create_corpus_df(df,1)

len(corpus_new1)
corpus_new1[:10]
# Generating the wordcloud with the values under the category dataframe

plt.figure(figsize=(12,8))

word_cloud = WordCloud(

                          background_color='black',

                          max_font_size = 80

                         ).generate(" ".join(corpus_new1[:50]))

plt.imshow(word_cloud)

plt.axis('off')

plt.show()
corpus_new0=create_corpus_df(df,0)

len(corpus_new0)
corpus_new0[:10]
# Generating the wordcloud with the values under the category dataframe

plt.figure(figsize=(12,8))

word_cloud = WordCloud(

                          background_color='black',

                          max_font_size = 80

                         ).generate(" ".join(corpus_new0[:50]))

plt.imshow(word_cloud)

plt.axis('off')

plt.show()
df.head(10)
def cv(data):

    count_vectorizer = CountVectorizer()



    emb = count_vectorizer.fit_transform(data)



    return emb, count_vectorizer



list_corpus = df["text"].tolist()

list_labels = df["target"].tolist()



X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, 

                                                                                random_state=40)



X_train_counts, count_vectorizer = cv(X_train)

X_test_counts = count_vectorizer.transform(X_test)
def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):

        lsa = TruncatedSVD(n_components=2)

        lsa.fit(test_data)

        lsa_scores = lsa.transform(test_data)

        color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}

        color_column = [color_mapper[label] for label in test_labels]

        colors = ['orange','blue']

        if plot:

            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))

            orange_patch = mpatches.Patch(color='orange', label='Not')

            blue_patch = mpatches.Patch(color='blue', label='Real')

            plt.legend(handles=[orange_patch, blue_patch], prop={'size': 30})



fig = plt.figure(figsize=(16, 16))          

plot_LSA(X_train_counts, y_train)

plt.show()
def tfidf(data):

    tfidf_vectorizer = TfidfVectorizer()



    train = tfidf_vectorizer.fit_transform(data)



    return train, tfidf_vectorizer



X_train_tfidf, tfidf_vectorizer = tfidf(X_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)
fig = plt.figure(figsize=(16, 16))          

plot_LSA(X_train_tfidf, y_train)

plt.show()
Dropout_num = 0

learning_rate = 5e-6

valid = 0.2

epochs_num = 3

batch_size_num = 8

target_corrected = True

target_big_corrected = True
# We will use the official tokenization script created by the Google team

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub



import tokenization
# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

def build_model(bert_layer, max_len=256):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    

    if Dropout_num == 0:

        # Without Dropout

        out = Dense(1, activation='sigmoid')(clf_output)

    else:

        # With Dropout(Dropout_num), Dropout_num > 0

        x = Dropout(Dropout_num)(clf_output)

        out = Dense(1, activation='sigmoid')(x)



    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def clean_tweets(tweet):

    """Removes links and non-ASCII characters"""

    

    tweet = ''.join([x for x in tweet if x in string.printable])

    

    # Removing URLs

    tweet = re.sub(r"http\S+", "", tweet)

    

    return tweet
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def remove_punctuations(text):

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"

    

    for p in punctuations:

        text = text.replace(p, f' {p} ')



    text = text.replace('...', ' ... ')

    

    if '...' not in text:

        text = text.replace('..', ' ... ')

    

    return text
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

abbreviations = {

    "$" : " dollar ",

    "€" : " euro ",

    "4ao" : "for adults only",

    "a.m" : "before midday",

    "a3" : "anytime anywhere anyplace",

    "aamof" : "as a matter of fact",

    "acct" : "account",

    "adih" : "another day in hell",

    "afaic" : "as far as i am concerned",

    "afaict" : "as far as i can tell",

    "afaik" : "as far as i know",

    "afair" : "as far as i remember",

    "afk" : "away from keyboard",

    "app" : "application",

    "approx" : "approximately",

    "apps" : "applications",

    "asap" : "as soon as possible",

    "asl" : "age, sex, location",

    "atk" : "at the keyboard",

    "ave." : "avenue",

    "aymm" : "are you my mother",

    "ayor" : "at your own risk", 

    "b&b" : "bed and breakfast",

    "b+b" : "bed and breakfast",

    "b.c" : "before christ",

    "b2b" : "business to business",

    "b2c" : "business to customer",

    "b4" : "before",

    "b4n" : "bye for now",

    "b@u" : "back at you",

    "bae" : "before anyone else",

    "bak" : "back at keyboard",

    "bbbg" : "bye bye be good",

    "bbc" : "british broadcasting corporation",

    "bbias" : "be back in a second",

    "bbl" : "be back later",

    "bbs" : "be back soon",

    "be4" : "before",

    "bfn" : "bye for now",

    "blvd" : "boulevard",

    "bout" : "about",

    "brb" : "be right back",

    "bros" : "brothers",

    "brt" : "be right there",

    "bsaaw" : "big smile and a wink",

    "btw" : "by the way",

    "bwl" : "bursting with laughter",

    "c/o" : "care of",

    "cet" : "central european time",

    "cf" : "compare",

    "cia" : "central intelligence agency",

    "csl" : "can not stop laughing",

    "cu" : "see you",

    "cul8r" : "see you later",

    "cv" : "curriculum vitae",

    "cwot" : "complete waste of time",

    "cya" : "see you",

    "cyt" : "see you tomorrow",

    "dae" : "does anyone else",

    "dbmib" : "do not bother me i am busy",

    "diy" : "do it yourself",

    "dm" : "direct message",

    "dwh" : "during work hours",

    "e123" : "easy as one two three",

    "eet" : "eastern european time",

    "eg" : "example",

    "embm" : "early morning business meeting",

    "encl" : "enclosed",

    "encl." : "enclosed",

    "etc" : "and so on",

    "faq" : "frequently asked questions",

    "fawc" : "for anyone who cares",

    "fb" : "facebook",

    "fc" : "fingers crossed",

    "fig" : "figure",

    "fimh" : "forever in my heart", 

    "ft." : "feet",

    "ft" : "featuring",

    "ftl" : "for the loss",

    "ftw" : "for the win",

    "fwiw" : "for what it is worth",

    "fyi" : "for your information",

    "g9" : "genius",

    "gahoy" : "get a hold of yourself",

    "gal" : "get a life",

    "gcse" : "general certificate of secondary education",

    "gfn" : "gone for now",

    "gg" : "good game",

    "gl" : "good luck",

    "glhf" : "good luck have fun",

    "gmt" : "greenwich mean time",

    "gmta" : "great minds think alike",

    "gn" : "good night",

    "g.o.a.t" : "greatest of all time",

    "goat" : "greatest of all time",

    "goi" : "get over it",

    "gps" : "global positioning system",

    "gr8" : "great",

    "gratz" : "congratulations",

    "gyal" : "girl",

    "h&c" : "hot and cold",

    "hp" : "horsepower",

    "hr" : "hour",

    "hrh" : "his royal highness",

    "ht" : "height",

    "ibrb" : "i will be right back",

    "ic" : "i see",

    "icq" : "i seek you",

    "icymi" : "in case you missed it",

    "idc" : "i do not care",

    "idgadf" : "i do not give a damn fuck",

    "idgaf" : "i do not give a fuck",

    "idk" : "i do not know",

    "ie" : "that is",

    "i.e" : "that is",

    "ifyp" : "i feel your pain",

    "IG" : "instagram",

    "iirc" : "if i remember correctly",

    "ilu" : "i love you",

    "ily" : "i love you",

    "imho" : "in my humble opinion",

    "imo" : "in my opinion",

    "imu" : "i miss you",

    "iow" : "in other words",

    "irl" : "in real life",

    "j4f" : "just for fun",

    "jic" : "just in case",

    "jk" : "just kidding",

    "jsyk" : "just so you know",

    "l8r" : "later",

    "lb" : "pound",

    "lbs" : "pounds",

    "ldr" : "long distance relationship",

    "lmao" : "laugh my ass off",

    "lmfao" : "laugh my fucking ass off",

    "lol" : "laughing out loud",

    "ltd" : "limited",

    "ltns" : "long time no see",

    "m8" : "mate",

    "mf" : "motherfucker",

    "mfs" : "motherfuckers",

    "mfw" : "my face when",

    "mofo" : "motherfucker",

    "mph" : "miles per hour",

    "mr" : "mister",

    "mrw" : "my reaction when",

    "ms" : "miss",

    "mte" : "my thoughts exactly",

    "nagi" : "not a good idea",

    "nbc" : "national broadcasting company",

    "nbd" : "not big deal",

    "nfs" : "not for sale",

    "ngl" : "not going to lie",

    "nhs" : "national health service",

    "nrn" : "no reply necessary",

    "nsfl" : "not safe for life",

    "nsfw" : "not safe for work",

    "nth" : "nice to have",

    "nvr" : "never",

    "nyc" : "new york city",

    "oc" : "original content",

    "og" : "original",

    "ohp" : "overhead projector",

    "oic" : "oh i see",

    "omdb" : "over my dead body",

    "omg" : "oh my god",

    "omw" : "on my way",

    "p.a" : "per annum",

    "p.m" : "after midday",

    "pm" : "prime minister",

    "poc" : "people of color",

    "pov" : "point of view",

    "pp" : "pages",

    "ppl" : "people",

    "prw" : "parents are watching",

    "ps" : "postscript",

    "pt" : "point",

    "ptb" : "please text back",

    "pto" : "please turn over",

    "qpsa" : "what happens", #"que pasa",

    "ratchet" : "rude",

    "rbtl" : "read between the lines",

    "rlrt" : "real life retweet", 

    "rofl" : "rolling on the floor laughing",

    "roflol" : "rolling on the floor laughing out loud",

    "rotflmao" : "rolling on the floor laughing my ass off",

    "rt" : "retweet",

    "ruok" : "are you ok",

    "sfw" : "safe for work",

    "sk8" : "skate",

    "smh" : "shake my head",

    "sq" : "square",

    "srsly" : "seriously", 

    "ssdd" : "same stuff different day",

    "tbh" : "to be honest",

    "tbs" : "tablespooful",

    "tbsp" : "tablespooful",

    "tfw" : "that feeling when",

    "thks" : "thank you",

    "tho" : "though",

    "thx" : "thank you",

    "tia" : "thanks in advance",

    "til" : "today i learned",

    "tl;dr" : "too long i did not read",

    "tldr" : "too long i did not read",

    "tmb" : "tweet me back",

    "tntl" : "trying not to laugh",

    "ttyl" : "talk to you later",

    "u" : "you",

    "u2" : "you too",

    "u4e" : "yours for ever",

    "utc" : "coordinated universal time",

    "w/" : "with",

    "w/o" : "without",

    "w8" : "wait",

    "wassup" : "what is up",

    "wb" : "welcome back",

    "wtf" : "what the fuck",

    "wtg" : "way to go",

    "wtpa" : "where the party at",

    "wuf" : "where are you from",

    "wuzup" : "what is up",

    "wywh" : "wish you were here",

    "yd" : "yard",

    "ygtr" : "you got that right",

    "ynk" : "you never know",

    "zzz" : "sleeping bored and tired"

}
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def convert_abbrev(word):

    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def convert_abbrev_in_text(text):

    tokens = word_tokenize(text)

    tokens = [convert_abbrev(word) for word in tokens]

    text = ' '.join(tokens)

    return text
# Load BERT from the Tensorflow Hub

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
# Load CSV files containing training data

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
# Thanks to https://www.kaggle.com/wrrosa/keras-bert-using-tfhub-modified-train-data - 

# author of this kernel read tweets in training data and figure out that some of them have errors:

# if target_corrected:

ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]

train.loc[train['id'].isin(ids_with_target_error),'target'] = 0

train[train['id'].isin(ids_with_target_error)]
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

# if target_big_corrected:

train["text"] = train["text"].apply(lambda x: clean_tweets(x))

test["text"] = test["text"].apply(lambda x: clean_tweets(x))



train["text"] = train["text"].apply(lambda x: remove_emoji(x))

test["text"] = test["text"].apply(lambda x: remove_emoji(x))



train["text"] = train["text"].apply(lambda x: remove_punctuations(x))

test["text"] = test["text"].apply(lambda x: remove_punctuations(x))



train["text"] = train["text"].apply(lambda x: convert_abbrev_in_text(x))

test["text"] = test["text"].apply(lambda x: convert_abbrev_in_text(x))
# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

# Load tokenizer from the bert layer

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

# Encode the text into tokens, masks, and segment flags

train_input = bert_encode(train.text.values, tokenizer, max_len=160)

test_input = bert_encode(test.text.values, tokenizer, max_len=160)

train_labels = train.target.values
# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

# Build BERT model with my tuning

model_BERT = build_model(bert_layer, max_len=160)

model_BERT.summary()
# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

# Train BERT model with my tuning

checkpoint = ModelCheckpoint('model_BERT.h5', monitor='val_loss', save_best_only=True)



train_history = model_BERT.fit(

    train_input, train_labels,

    validation_split = valid,

    epochs = epochs_num, # recomended 3-5 epochs

    callbacks=[checkpoint],

    batch_size = batch_size_num

)
# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

# Prediction by BERT model with my tuning

model_BERT.load_weights('model_BERT.h5')

test_pred_BERT = model_BERT.predict(test_input)

test_pred_BERT_int = test_pred_BERT.round().astype('int')
# Prediction by BERT model with my tuning for the training data - for the Confusion Matrix

train_pred_BERT = model_BERT.predict(train_input)

train_pred_BERT_int = train_pred_BERT.round().astype('int')
pred = pd.DataFrame(test_pred_BERT, columns=['preds'])

pred.plot.hist()
submission['target'] = test_pred_BERT_int

submission.head(10)
submission.to_csv("submission.csv", index=False, header=True)
# Showing Confusion Matrix

def plot_cm(y_true, y_pred, title, figsize=(5,4)):

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)

    plt.title(title)

    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
# Showing Confusion Matrix for GloVe model

# plot_cm(train_pred_GloVe_int, train['target'].values, 'Confusion matrix for GloVe model', figsize=(7,7))
# Showing Confusion Matrix for BERT model

# plot_cm(train_pred_BERT_int, train['target'].values, 'Confusion matrix for BERT model', figsize=(7,7))