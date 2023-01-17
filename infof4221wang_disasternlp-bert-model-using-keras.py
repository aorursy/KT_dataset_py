# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Changing the number of characters displayed in pandas 

pd.options.display.max_colwidth = 150
import seaborn as sns

# preprocessing

import regex as re

import string

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from tqdm import tqdm

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

# data split

from sklearn.model_selection import train_test_split

# model

import tensorflow as tf

from keras.models import Sequential, Model

from keras.layers import Embedding, LSTM,Dense, SpatialDropout1D, Dropout, Input

from keras.initializers import Constant

from keras.optimizers import Adam

# bert

!pip install -q tf-models-official==2.3.0

from official.modeling import tf_utils

from official import nlp

from official.nlp import bert

import tensorflow_hub as hub

from tensorflow.keras.callbacks import ModelCheckpoint



import official.nlp.bert.tokenization as tokenization
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')

submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
train.info()

test.info()
train.head(10)
sns.countplot(data=train, x='target')
train['words'] = train['text'].str.split()

train['word_len'] = train['words'].map(lambda x: len(x))

train['word_len'].max()
test['words'] = test['text'].str.split()

test['word_len'] = test['words'].map(lambda x: len(x))

test['word_len'].max()
# Thanks to https://www.kaggle.com/wrrosa/keras-bert-using-tfhub-modified-train-data - 

# author of this kernel read tweets in training data and figure out that some of them have errors:

ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]

train.loc[train['id'].isin(ids_with_target_error),'target'] = 0

train[train['id'].isin(ids_with_target_error)]
df_all = pd.concat([train,test])

df_all.shape
df_all['text'] = df_all['text'].str.lower()

df_all['text'].head(2)
def remove_breaklines(text):

    return re.sub('\n','',text)

df_all['text'] = df_all['text'].apply(lambda x: remove_breaklines(x))
def remove_numbers(text):

    return re.sub('\w*\d\w*', '', text)

df_all['text'] = df_all['text'].apply(lambda x: remove_numbers(x))
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



df_all['text'] = df_all['text'].apply(lambda x: remove_URL(x))
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

df_all['text']=df_all['text'].apply(lambda x : remove_html(x))
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



df_all['text']=df_all['text'].apply(lambda x: remove_emoji(x))
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



df_all['text']=df_all['text'].apply(lambda x : remove_punct(x))


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

    "qpsa" : "what happens", 

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

    return abbreviations[word] if word in abbreviations.keys() else word
def convert_abbrev_in_text(text):

    tokens = word_tokenize(text)

    tokens = [convert_abbrev(word) for word in tokens]

    text = ' '.join(tokens)

    return text



df_all['text']=df_all['text'].apply(lambda x : convert_abbrev_in_text(x))
df_all['text']=df_all['text'].apply(lambda x : word_tokenize(x))
stop = set(stopwords.words('english'))

def remove_stopwords(text):

    words = [w for w in text if w not in stop]

    return ' '.join(words)



df_all['text']=df_all['text'].apply(lambda x : remove_stopwords(x))
cl_ch_len = df_all['text'].apply(lambda x: len(x))

cl_wd_len = df_all['text'].str.split().map(lambda x: len(x))

print('Max words length for cleaned tweets: {}'.format(max(cl_wd_len)))

print('Max characters length for cleaned tweets: {}'.format(max(cl_ch_len)))

MAX_LEN = max(cl_wd_len)
def create_corpus(df):

    corpus=[]

    # tqdm show progress bar

    for tweet in tqdm(df_all['text']):

        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]

        corpus.append(words)

    return corpus



corpus = create_corpus(df_all)

corpus[0]
glove_embedding_dict={}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt','r') as f:

    for line in tqdm(f):

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        glove_embedding_dict[word]=vectors
# Define the dimension of word embeddings

W_E_DIM = 200
tokenizer_obj=Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

# Transforms each text in texts to a sequence of integers.

sequences=tokenizer_obj.texts_to_sequences(corpus)

# Padding

tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))
num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,W_E_DIM))



for word,i in tqdm(word_index.items()):

    if i < num_words:

        emb_vec=glove_embedding_dict.get(word)

        if emb_vec is not None:

            embedding_matrix[i]=emb_vec 
train_text = tweet_pad[:train.shape[0]]

test_text = tweet_pad[train.shape[0]:]

X_train,X_dev,Y_train,Y_dev=train_test_split(train_text,train['target'].values,test_size=0.2)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_dev.shape)
model=Sequential()

# The Embedding layer can be understood as a lookup table that maps from integer indices  to their embeddings. 

embedding=Embedding(num_words,W_E_DIM,embeddings_initializer=Constant(embedding_matrix),

                   input_length=MAX_LEN,trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))





optimzer=Adam(learning_rate=3e-4)



model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])



model.summary()
history=model.fit(X_train,Y_train,batch_size=4,

                  epochs=10,validation_data=(X_dev,Y_dev),verbose=2)

pred = model.predict(test_text)

pred = pred.round().astype('int')
df_sub = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

df_sub['target'] = pred

df_sub = df_sub[['id', 'target']]

df_sub.to_csv('lstm_submission.csv', index=False, header=True)

df_sub.head(10)
%%time

bert_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2'

bert_layer = hub.KerasLayer(bert_url, trainable=True, name='Bert_layer')
def bert_encode(texts, tokenizer, max_len=512):

    # build three inputs embeddings

    input_tokens = [] # token embedding 

    input_masks = [] # mask embedding, mask padding as 0

    input_segments = [] # segment embedding, segment two sentences

    

    for text in texts:

        text = tokenizer.tokenize(text)

        # segment 

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        # padding

        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        # pad_mask is 0 for padding and 1 for other

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        # single sentence segment_ids are all 0

        segment_ids = [0] * max_len

        

        input_tokens.append(tokens)

        input_masks.append(pad_masks)

        input_segments.append(segment_ids)

    

    return np.array(input_tokens), np.array(input_masks), np.array(input_segments)
def build_model(bert_layer, max_len=512):

    # build up input layer

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    # define bert layer outputs:[batch_size, 1024] and [batch_size, max_len, 1024]

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    # take [CLS](first token) in the output as classification features

    cls_output = sequence_output[:,0,:]

    output = Dense(1, activation='sigmoid')(cls_output)

    # define model

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=output)

    model.compile(Adam(lr=1e-5), loss = 'binary_crossentropy',metrics=['accuracy'])

    

    return model

    
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_text = df_all[:train.shape[0]].text

test_text = df_all[train.shape[0]:].text

train_input = bert_encode(train_text, tokenizer, max_len=160)

test_input = bert_encode(test_text, tokenizer, max_len = 160)

train_labels = train.target.values
model = build_model(bert_layer, max_len=160)

model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)



train_history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=5,

    callbacks=[checkpoint],

    batch_size=16

)
model.load_weights('model.h5')

test_pred = model.predict(test_input)
submission['target'] = test_pred.round().astype(int)

submission.to_csv('bert_submission.csv', index=False)