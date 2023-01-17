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
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

for dirname, _, filenames in os.walk('.'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import re

import string

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



def convert_abbrev(word):

    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word



# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

from nltk.tokenize import word_tokenize

def convert_abbrev_in_text(text):

    tokens = word_tokenize(text)

    tokens = [convert_abbrev(word) for word in tokens]

    text = ' '.join(tokens)

    return text



def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



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



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)

# version 15

convert_abbrev_flag = False

model_type = 'default'

Dropout_num = 0

learning_rate = 1e-5

max_len = 160

layers = [] #not including final layer

activation = 'relu' #for the non-final layers

remove_emoji_flag = False

remove_URL_flag = False

remove_html_flag = False

remove_punct_flag = False

epochs=3

batch_size=16

validation_split=0.2

remove_rows_based_on_cv = True #remove rows based on cross val predictions from train set

remove_row_cutoffs = (0.4,0.6) #if cutoffs for class 0 and 1

model_name = 'bert_default_dropout_{}_shape_{}'.format(Dropout_num,'_'.join([str(i) for i in layers]))
# # version 14

# convert_abbrev_flag = False

# model_type = 'default'

# Dropout_num = 0

# learning_rate = 1e-5

# max_len = 160

# layers = [] #not including final layer

# activation = 'relu' #for the non-final layers

# remove_emoji_flag = False

# remove_URL_flag = False

# remove_html_flag = False

# remove_punct_flag = False

# epochs=3

# batch_size=16

# validation_split=0.2

# remove_rows_based_on_cv = True #remove rows based on cross val predictions from train set

# remove_row_cutoffs = (0.5,0.5) #if cutoffs for class 0 and 1

# model_name = 'bert_default_dropout_{}_shape_{}'.format(Dropout_num,'_'.join([str(i) for i in layers]))
# # version 13

# convert_abbrev_flag = False

# model_type = 'default'

# Dropout_num = 0

# learning_rate = 1e-5

# max_len = 160

# layers = [] #not including final layer

# activation = 'relu' #for the non-final layers

# remove_emoji_flag = False

# remove_URL_flag = False

# remove_html_flag = False

# remove_punct_flag = True

# epochs=3

# batch_size=16

# validation_split=0.2

# remove_rows_based_on_cv = True #remove rows based on cross val predictions from train set

# model_name = 'bert_default_dropout_{}_shape_{}'.format(Dropout_num,'_'.join([str(i) for i in layers]))
# # version 12 - best till now

# convert_abbrev_flag = False

# model_type = 'default'

# Dropout_num = 0

# learning_rate = 1e-5

# max_len = 160

# layers = [] #not including final layer

# activation = 'relu' #for the non-final layers

# remove_emoji_flag = False

# remove_URL_flag = False

# remove_html_flag = False

# remove_punct_flag = False

# epochs=3

# batch_size=16

# validation_split=0.2

# remove_rows_based_on_cv = True #remove rows based on cross val predictions from train set

# model_name = 'bert_default_dropout_{}_shape_{}'.format(Dropout_num,'_'.join([str(i) for i in layers]))
# #version 10

# convert_abbrev_flag = False

# model_type = 'default'

# Dropout_num = 0

# learning_rate = 1e-6

# max_len = 200

# layers = [] #not including final layer

# activation = 'relu' #for the non-final layers

# remove_emoji_flag = False

# remove_URL_flag = False

# remove_html_flag = False

# remove_punct_flag = False

# epochs=3

# batch_size=16

# validation_split=0.2

# model_name = 'bert_default_dropout_{}_shape_{}'.format(Dropout_num,'_'.join([str(i) for i in layers]))
# #version 8

# convert_abbrev_flag = False

# model_type = 'default'

# Dropout_num = 0

# learning_rate = 1e-5

# max_len = 200

# layers = [] #not including final layer

# activation = 'relu' #for the non-final layers

# remove_emoji_flag = True

# remove_URL_flag = True

# remove_html_flag = True

# remove_punct_flag = True

# model_name = 'bert_default_preprocessed_dropout_{}_shape_{}'.format(Dropout_num,'_'.join([str(i) for i in layers]))
# #version 7

# convert_abbrev_flag = False

# model_type = 'default'

# Dropout_num = 0

# learning_rate = 1e-5

# max_len = 200

# layers = [30] #not including final layer

# activation = 'relu' #for the non-final layers

# model_name = 'model_bert_default_dropout_{}_shape_{}'.format(Dropout_num,'_'.join([str(i) for i in layers]))
# #version 6

# convert_abbrev = False

# model_type = 'dropout'

# Dropout_num = 0.1

# learning_rate = 1e-5

# max_len = 200

# layers = [30] #not including final layer

# activation = 'relu' #for the non-final layers

# model_name = 'model_bert_dropout_{}_shape_{}'.format(Dropout_num,'_'.join([str(i) for i in layers]))
# version 5

# convert_abbrev = False

# model_type = 'Dropout'

# Dropout_num = 0

# learning_rate = 1e-5

# max_len = 200

# model_name = 'model_bert_GlobalAveragePooling1D'

# layers = [30] #not including final layer

# activation = 'relu' #for the non-final layers
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
import pickle

with open('../input/bert-prediction/predictions.pkl','rb') as f:

    predictions = pickle.load(f)
train['infold_pred'] = predictions['infold_pred']

train['outfold_pred'] = predictions['outfold_pred']




if convert_abbrev_flag:

    train["text"] = train["text"].apply(lambda x: convert_abbrev_in_text(x))

    test["text"] = test["text"].apply(lambda x: convert_abbrev_in_text(x))

    

if remove_emoji_flag:

    train["text"] = train["text"].apply(lambda x: remove_emoji(x))

    test["text"] = test["text"].apply(lambda x: remove_emoji(x))



if remove_URL_flag:

    train['text'] = train['text'].apply(remove_URL)

    test['text'] = test['text'].apply(remove_URL)

if remove_html_flag:

    train['text'] = train['text'].apply(remove_html)

    test['text'] = test['text'].apply(remove_html)

if remove_punct_flag:

    train['text'] = train['text'].apply(remove_punct)

    test['text'] = test['text'].apply(remove_punct)
print(train.shape)

if remove_rows_based_on_cv:

    train = train[((train['target']==0)&(train['outfold_pred']<remove_row_cutoffs[0]))|((train['target']==1)&(train['outfold_pred']>remove_row_cutoffs[1]))]

print(train.shape)
# We will use the official tokenization script created by the Google team

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input,Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub



import tokenization
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

def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    

    if model_type=='default':

        # Without Dropout

        for layer in layers:

            clf_output = Dense(layer, activation=activation)(clf_output)

        out = Dense(1, activation='sigmoid')(clf_output)

    elif model_type=='dropout':

        # With Dropout(Dropout_num), Dropout_num > 0

        for layer in layers:

            x = Dropout(Dropout_num)(clf_output)

            clf_output = Dense(layer, activation=activation)(x)

        x = Dropout(Dropout_num)(clf_output)

        out = Dense(1, activation='sigmoid')(x)

    elif model_type=='GlobalAveragePooling1D':

        for layer in layers:

            if Dropout_num>0:

                clf_output = Dropout(Dropout_num)(clf_output)

            clf_output = Dense(layer, activation=activation)(clf_output)

        x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)

        out = Dense(1, activation='sigmoid')(x)



    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model

%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"#/2 is the updated version. need to try with that

# module_url = "https://tfhub.dev/tensorflow/albert_en_base/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encode(train.text.values, tokenizer, max_len=max_len)

test_input = bert_encode(test.text.values, tokenizer, max_len=max_len)

train_labels = train.target.values



print(train_input[0].shape,test_input[0].shape)
model = build_model(bert_layer, max_len=max_len)

model.summary()
%%time

checkpoint = ModelCheckpoint('{}.h5'.format(model_name), monitor='val_loss', save_best_only=True)



train_history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=epochs,

    callbacks=[checkpoint],

    batch_size=batch_size,

)
model.load_weights('{}.h5'.format(model_name))

test_pred = model.predict(test_input)
submission['target'] = test_pred.round().astype(int)

submission.to_csv('{}.csv'.format(model_name), index=False)
os.remove('{}.h5'.format(model_name))