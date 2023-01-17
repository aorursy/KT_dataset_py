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
tweet= pd.read_csv('../input/nlpgettingstarted/train.csv')

test=pd.read_csv('../input/nlpgettingstarted/test.csv')

submission = pd.read_csv("../input/nlpgettingstarted/sample_submission.csv")
#61

# Recomended 10-20 epochs

history=model.fit(X_train,y_train,batch_size=4,epochs=10,validation_data=(X_test,y_test),verbose=2)
#62

train_pred_GloVe = model.predict(train)

train_pred_GloVe_int = train_pred_GloVe.round().astype('int')
#63

# We will use the official tokenization script created by the Google team

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tokenization

import tensorflow_hub as hub

#64

import pandas as pd

import numpy as np

import os



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns



import nltk

nltk.download('stopwords')

nltk.download('punkt')

#C:\Users\Administrator\AppData\Roaming\nltk_data\corpora

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

import  tensorflow as tf

from    tensorflow import keras

'''

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding, LSTM,Dense, SpatialDropout1D, Dropout

from keras.initializers import Constant

from keras.optimizers import Adam

'''

Tokenizer = keras.preprocessing.text.Tokenizer

pad_sequences = keras.preprocessing.sequence.pad_sequences

Sequential = keras.models.Sequential

Embedding = keras.layers.Embedding

LSTM = keras.layers.LSTM

Dense = keras.layers.Dense

SpatialDropout1D = keras.layers.SpatialDropout1D

Dropout = keras.layers.Dropout

Constant = keras.initializers.Constant

Adam = keras.optimizers.Adam



Dropout_num = 0

learning_rate = 6e-6

valid = 0.2

epochs_num = 3

batch_size_num = 16

target_corrected = True

target_big_corrected = False





Input = keras.layers.Input

Model = keras.models.Model

ModelCheckpoint = keras.callbacks.ModelCheckpoint
#65

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
#66

# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

def build_model(bert_layer, max_len=512):

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
#67

# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def clean_tweets(tweet):

    """Removes links and non-ASCII characters"""

    

    tweet = ''.join([x for x in tweet if x in string.printable])

    

    # Removing URLs

    tweet = re.sub(r"http\S+", "", tweet)

    

    return tweet
#68

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
#69

# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def remove_punctuations(text):

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"

    

    for p in punctuations:

        text = text.replace(p, f' {p} ')



    text = text.replace('...', ' ... ')

    

    if '...' not in text:

        text = text.replace('..', ' ... ')

    

    return text
#70

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
#71

# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def convert_abbrev(word):

    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word
#72

# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def convert_abbrev_in_text(text):

    tokens = word_tokenize(text)

    tokens = [convert_abbrev(word) for word in tokens]

    text = ' '.join(tokens)

    return text
#73

# Load BERT from the Tensorflow Hub

#module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

#module_url = "https://www.colossusjinxin.com/1"

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
#74

# Load CSV files containing training data

train = pd.read_csv("../input/nlpgettingstarted/train.csv")

test = pd.read_csv("../input/nlpgettingstarted/test.csv")
#75

# Thanks to https://www.kaggle.com/wrrosa/keras-bert-using-tfhub-modified-train-data - 

# author of this kernel read tweets in training data and figure out that some of them have errors:

if target_corrected:

    ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]

    train.loc[train['id'].isin(ids_with_target_error),'target'] = 0

    train[train['id'].isin(ids_with_target_error)]
#76

# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

if target_big_corrected:

    train["text"] = train["text"].apply(lambda x: clean_tweets(x))

    test["text"] = test["text"].apply(lambda x: clean_tweets(x))

    

    train["text"] = train["text"].apply(lambda x: remove_emoji(x))

    test["text"] = test["text"].apply(lambda x: remove_emoji(x))

    

    train["text"] = train["text"].apply(lambda x: remove_punctuations(x))

    test["text"] = test["text"].apply(lambda x: remove_punctuations(x))

    

    train["text"] = train["text"].apply(lambda x: convert_abbrev_in_text(x))

    test["text"] = test["text"].apply(lambda x: convert_abbrev_in_text(x))
#77

# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

# Load tokenizer from the bert layer

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
#78

# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

# Encode the text into tokens, masks, and segment flags

train_input = bert_encode(train.text.values, tokenizer, max_len=160)

test_input = bert_encode(test.text.values, tokenizer, max_len=160)

train_labels = train.target.values
#79

# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

# Build BERT model with my tuning

model_BERT = build_model(bert_layer, max_len=160)

model_BERT.summary()

#80

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
#81

# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

# Prediction by BERT model with my tuning

model_BERT.load_weights('model_BERT.h5')

test_pred_BERT = model_BERT.predict(test_input)

test_pred_BERT_int = test_pred_BERT.round().astype('int')
#82

# Prediction by BERT model with my tuning for the training data - for the Confusion Matrix

train_pred_BERT = model_BERT.predict(train_input)

train_pred_BERT_int = train_pred_BERT.round().astype('int')
#83

pred = pd.DataFrame(test_pred_BERT, columns=['preds'])

pred.plot.hist()
#84

submission = pd.read_csv("../input/nlpgettingstarted/sample_submission.csv")

submission['target'] = test_pred_BERT_int

submission.head(10)
#85

submission.to_csv("submission.csv", index=False, header=True)
#86

# Showing Confusion Matrix

def plot_cm(y_true, y_pred, title, figsize=(5,5)):

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
#88

# Showing Confusion Matrix for BERT model

plot_cm(train_pred_BERT_int, train['target'].values, 'Confusion matrix for BERT model', figsize=(7,7))