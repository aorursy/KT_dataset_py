import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Conv1D, GlobalMaxPooling1D, MaxPooling1D, LeakyReLU,Dropout,AvgPool2D, UpSampling2D, ReLU, MaxPooling2D, Reshape, Softmax, Activation, Flatten, Lambda, Conv2DTranspose
from tensorflow.keras.losses import MSE, categorical_crossentropy, binary_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

import time 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import plotly.graph_objects as go
import re

import nltk  
nltk.download('stopwords') 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from collections import Counter
import cufflinks as cf
cf.go_offline()
from sklearn.utils import shuffle  


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
submission =  pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train.head()
train.info()
def Plot_world(text):
    
    comment_words = ' '
    stopwords = set(STOPWORDS) 
    
    for val in text: 

        # typecaste each val to string 
        val = str(val) 

        # split the value 
        tokens = val.split() 

        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 

        for words in tokens: 
            comment_words = comment_words + words + ' '


    wordcloud = WordCloud(width = 5000, height = 4000, 
                    background_color ='black', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words) 

    # plot the WordCloud image                        
    plt.figure(figsize = (12, 12), facecolor = 'k', edgecolor = 'k' ) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.show()
text = train.text.values
Plot_world(text)
train.loc[train['text'].str.contains('http')].target.value_counts()
keywords = train.keyword.values
Plot_world(keywords)
targ_zero = 0
targ_one = 0
for ii in range(len(train.target)):
    if train.target[ii] == 1:
        targ_one = targ_one+1
    elif train.target[ii] == 0:
        targ_zero = targ_zero + 1
        
print(f"There are {targ_zero} tweets with target 0 and {targ_one} tweets with target 1")
pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

def remove_html(text):
    no_html= pattern.sub('',text)
    return no_html

train['text']=train['text'].apply(lambda x : remove_html(x))
test['text']=test['text'].apply(lambda x : remove_html(x))
train.loc[train['text'].str.contains('http')].target.value_counts()

text = 'Make sure to check out our 5LSL0 Kaggle competition contribution! https://www.kaggle.com/c/nlp-getting-started'
print('Original: '+ text)
print('Removed URL: '+ remove_html(text))
def clean_text(text):
 
    text = re.sub('[^a-zA-Z]', ' ', text)  # means any character that IS NOT a-z OR A-Z 
    
    # re.compile is used to save the resulting regular expression object
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    text = text.lower()  # To convert to lower case

    # split to array(default delimiter is " ") 
    text = text.split()  

    text = ' '.join(text)    
            
    return text

train['text'] = train['text'].apply(lambda x : clean_text(x))
test['text']=test['text'].apply(lambda x : clean_text(x))

text = 'After using the regularization method dropout, we increased our accuracy by 5%!!! :)'
print('Original: '+ text)
print('Cleaned: ' + clean_text(text))
def decontracted(phrase):
    
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    
    return phrase

train['text'] = train['text'].apply(lambda x : decontracted(x))
test['text']=test['text'].apply(lambda x : decontracted(x))

text = 'We\'re proud that our algorithm\'ll save lives in the future!'
print('Original: '+text)
print('Decontracted: '+ decontracted(text))
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

def convert_abbrev_in_text(text):
    t=[]
    words=text.split()
    t = [abbreviations[w.lower()] if w.lower() in abbreviations.keys() else w for w in words]
    return ' '.join(t)    

train['text'] = train['text'].apply(lambda x : convert_abbrev_in_text(x))
test['text']=test['text'].apply(lambda x : convert_abbrev_in_text(x))

text = 'Long short-term memory (LSTM) networks are the wtg'

print('Original: '+text)
print('Decontracted: '+convert_abbrev_in_text(text))
white_list = {'no', 'not'}
stop_words = set(stopwords.words('english'))

final_stop_words = set([word for word in stop_words if word not in white_list])

print(train['text'][40])
train['tokens'] = train.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
test['tokens'] = test.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)

train['tokens']=train['tokens'].apply(lambda row: [word for word in row if word not in final_stop_words])
test['tokens']=test['tokens'].apply(lambda row: [word for word in row if word not in final_stop_words])

train['text'] = train.apply( lambda row: " ".join(row['tokens']), axis=1)
test['text'] = test.apply( lambda row: " ".join(row['tokens']), axis=1)

print(train['tokens'][40])
print(train['text'][40])
def counter_word (text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count

text_values = train["text"]
counter = counter_word(text_values)
print(f"The len of unique words is: {len(counter)}")
max_len = 0
for ii in range(len(train.tokens)):
    if len(train.tokens[ii]) > max_len:
        max_len = len(train.tokens[ii])

print(f"The longest tweet contains {max_len} words")
vocab_size = len(counter)

max_length = 25
trunc_type='post'
padding_type='post'

oov_tok = "<XXX>"
val_split = 0.2
training_size = round((1-val_split)*len(train))

train = shuffle(train) 
training_sentences = train.text[0:training_size]
training_labels = train.target[0:training_size]

testing_sentences = train.text[training_size:]
testing_labels = train.target[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

print("The first word indices are: ")
for x in list(word_index)[1:15]:
    print (" {},  {} ".format(x,  word_index[x]))
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(train.text[0])
print(testing_padded[0])
from pathlib import Path

GLOVE_DIM = 200  # Number of dimensions of the GloVe word embeddings

root = Path('../')
input_path = root / 'input/' 

glove_file = 'glove.twitter.27B.' + str(GLOVE_DIM) + 'd.txt'
glove_dir = 'glove-global-vectors-for-word-representation/'
emb_dict = {}
glove = open(input_path / glove_dir / glove_file)
for line in glove:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    emb_dict[word] = vector
glove.close()

emb_matrix = np.zeros((vocab_size, GLOVE_DIM))

for w, i in word_index.items():
    if i < vocab_size:
        vect = emb_dict.get(w)
        # Check if the word from the training data occurs in the GloVe word embeddings
        # Otherwise the vector is kept with only zeros. 
        if vect is not None:
            emb_matrix[i] = vect
    else:
        break
def plot_loss_acc(model_hist):
    plt.subplot(1,2,1)
    plt.plot(model_hist.history['loss'])
    plt.plot(model_hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(b='true')
    plt.legend(['Training loss', 'Validation loss'], loc='best')

    plt.subplot(1,2,2)
    plt.plot(model_hist.history['accuracy'])
    plt.plot(model_hist.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(b='true')
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='best')
    plt.tight_layout()
    plt.show()
keras.backend.clear_session()
DC_model_1 = Sequential()
DC_model_1.add(Embedding(vocab_size, GLOVE_DIM, input_length=max_length, weights =[emb_matrix], trainable=True))
DC_model_1.add(Conv1D(32, 5, activation='relu', padding='same')) 
DC_model_1.add(GlobalMaxPooling1D())
DC_model_1.add(Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam(lr=0.0001)
DC_model_1.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
DC_model_1.summary()

#Train the model
num_epochs = 20
start_time = time.time()

DC_history = DC_model_1.fit(training_padded, training_labels, \
                        epochs=num_epochs, validation_data=(testing_padded, testing_labels), \
                        shuffle=True, \
                        verbose=1)

 

print(f'The time in seconds: {(time.time()- start_time)}')

#Plot results
plot_loss_acc(DC_history)
keras.backend.clear_session()
DC_model_2 = Sequential()
DC_model_2.add(Embedding(vocab_size, GLOVE_DIM, input_length=max_length, weights =[emb_matrix], trainable=True))
DC_model_2.add(Dropout(0.3))
DC_model_2.add(Conv1D(32, 5, activation='relu', padding='same')) 
DC_model_2.add(GlobalMaxPooling1D())
DC_model_2.add(Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam(lr=0.0001)
DC_model_2.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
DC_model_2.summary()

#Train the model
num_epochs = 20
start_time = time.time()
earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=1, mode='auto')

DC_ESDO_history = DC_model_2.fit(training_padded, training_labels, \
                        epochs=num_epochs, validation_data=(testing_padded, testing_labels), \
                        shuffle=True, \
                        callbacks=[earlystop],
                        verbose=1)

 

print(f'The time in seconds: {(time.time()- start_time)}')

#Plot results
plot_loss_acc(DC_ESDO_history)
keras.backend.clear_session()
DC_model_3 = Sequential()
DC_model_3.add(Embedding(vocab_size, GLOVE_DIM, input_length=max_length, weights =[emb_matrix], trainable=True))
DC_model_3.add(Conv1D(32, 5, kernel_regularizer=regularizers.l2(0.05), activation='relu', padding='same')) 
DC_model_3.add(GlobalMaxPooling1D())
DC_model_3.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.05)))

optimizer = tf.keras.optimizers.Adam(lr=0.0001)

DC_model_3.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
DC_model_3.summary()

#Train the model
num_epochs = 20
start_time = time.time()
earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=1, mode='auto')

DC_ESL2_history = DC_model_3.fit(training_padded, training_labels, \
                        epochs=num_epochs, validation_data=(testing_padded, testing_labels), \
                        shuffle=True, \
                        callbacks=[earlystop],
                        verbose=1)

 

print(f'The time in seconds: {(time.time()- start_time)}')

#Plot results
plot_loss_acc(DC_ESL2_history)
plot_loss_acc(DC_history)
plot_loss_acc(DC_ESDO_history)
plot_loss_acc(DC_ESL2_history)
keras.backend.clear_session()
LSTM_model_1 = Sequential()
LSTM_model_1.add(Embedding(vocab_size, GLOVE_DIM, input_length=max_length, weights =[emb_matrix], trainable=True))
LSTM_model_1.add(LSTM(16))
LSTM_model_1.add(Dense(8, activation='tanh'))
LSTM_model_1.add(Dense(1, activation='sigmoid'))  

optimizer = tf.keras.optimizers.Adam(lr=0.0001)

LSTM_model_1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
num_epochs = 20
start_time = time.time()

LSTM_history_1 = LSTM_model_1.fit(training_padded, training_labels, \
                        epochs=num_epochs, validation_data=(testing_padded, testing_labels), \
                        shuffle=True,
                        verbose=0)

plot_loss_acc(LSTM_history_1)
keras.backend.clear_session()
LSTM_model_2 = Sequential()
LSTM_model_2.add(Embedding(vocab_size, GLOVE_DIM, input_length=max_length, weights =[emb_matrix], trainable=True))
LSTM_model_2.add(LSTM(16, recurrent_dropout=0.3))
LSTM_model_2.add(Dense(8, activation='tanh', kernel_regularizer=regularizers.l2(0.05)))
LSTM_model_2.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.05)))  

optimizer = tf.keras.optimizers.Adam(lr=0.0001)

LSTM_model_2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
num_epochs = 30
start_time = time.time()
earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=1, mode='auto')

LSTM_history_2 = LSTM_model_2.fit(training_padded, training_labels, \
                        epochs=num_epochs, validation_data=(testing_padded, testing_labels), \
                        shuffle=True, \
                        callbacks=[earlystop],
                        verbose=0)

plot_loss_acc(LSTM_history_2)
keras.backend.clear_session()
LSTM_model_3 = Sequential()
LSTM_model_3.add(Embedding(vocab_size, GLOVE_DIM, input_length=max_length, weights =[emb_matrix], trainable=True))
LSTM_model_3.add(LSTM(16, dropout=0.3, recurrent_dropout=0.3))
LSTM_model_3.add(Dense(8, activation='tanh'))
LSTM_model_3.add(Dense(1, activation='sigmoid'))  

optimizer = tf.keras.optimizers.Adam(lr=0.0001)

LSTM_model_3.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
num_epochs = 20
start_time = time.time()
earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=1, mode='auto')

LSTM_history_3 = LSTM_model_3.fit(training_padded, training_labels, \
                        epochs=num_epochs, validation_data=(testing_padded, testing_labels), \
                        shuffle=True, \
                        callbacks=[earlystop],
                        verbose=0)

plot_loss_acc(LSTM_history_3)
# calculating metrics for a neural network model using sklearn


# Calculating predictions from LSTM and CNN model(s)
predictions_DCmodel_1 = DC_model_1.predict_classes(testing_padded)
predictions_DCmodel_2 = DC_model_2.predict_classes(testing_padded)
predictions_DCmodel_3 = DC_model_3.predict_classes(testing_padded)

predictions_LSTM_model_1 = LSTM_model_1.predict_classes(testing_padded)
predictions_LSTM_model_2 = LSTM_model_2.predict_classes(testing_padded)
predictions_LSTM_model_3 = LSTM_model_3.predict_classes(testing_padded)


# Calculating accuracy for each models

# For CNN and LSTM 
# accuracy: (tp + tn) / (p + n)
accuracy_CNN = accuracy_score(testing_labels, predictions_DCmodel_1)
accuracy_LSTM = accuracy_score(testing_labels, predictions_LSTM_model_1)
print('Without regularization or dropout, accuracy of CNN = %f, and accuracy of LSTM = %f' % (accuracy_CNN, accuracy_LSTM))

accuracy_CNN = accuracy_score(testing_labels, predictions_DCmodel_2)
accuracy_LSTM = accuracy_score(testing_labels, predictions_LSTM_model_2)
print('With early stopping and dropout, accuracy of CNN = %f, and accuracy of LSTM = %f' % (accuracy_CNN, accuracy_LSTM))

accuracy_CNN = accuracy_score(testing_labels, predictions_DCmodel_3)
accuracy_LSTM = accuracy_score(testing_labels, predictions_LSTM_model_3)
print('With early stopping and regularization, accuracy of CNN = %f, and accuracy of LSTM = %f' % (accuracy_CNN, accuracy_LSTM))


# Calculating precision for each models

# For CNN
# precision tp / (tp + fp)
precision = precision_score(testing_labels, predictions_DCmodel_1)
#print('Precision without regularization or dropout: %f' % precision)

precision = precision_score(testing_labels, predictions_DCmodel_2)
#print('Precision with early stopping and dropout: %f' % precision)

precision = precision_score(testing_labels, predictions_DCmodel_3)
#print('Precision with early stopping and regularization: %f' % precision)


# For LSTM
# precision tp / (tp + fp)
precision = precision_score(testing_labels, predictions_LSTM_model_1)
#print('Precision without regularization or dropout: %f' % precision)

precision = precision_score(testing_labels, predictions_LSTM_model_2)
#print('Precision with early stopping and dropout: %f' % precision)

precision = precision_score(testing_labels, predictions_LSTM_model_3)
#print('Precision with early stopping and regularization: %f' % precision)

# Calculating precision for each models

# For CNN
# precision tp / (tp + fp)
precision = precision_score(testing_labels, predictions_DCmodel_1)
#print('Precision without regularization or dropout: %f' % precision)

precision = precision_score(testing_labels, predictions_DCmodel_2)
#print('Precision with early stopping and dropout: %f' % precision)

precision = precision_score(testing_labels, predictions_DCmodel_3)
#print('Precision with early stopping and regularization: %f' % precision)


# For LSTM
# precision tp / (tp + fp)
precision = precision_score(testing_labels, predictions_LSTM_model_1)
#print('Precision without regularization or dropout: %f' % precision)

precision = precision_score(testing_labels, predictions_LSTM_model_2)
#print('Precision with early stopping and dropout: %f' % precision)

precision = precision_score(testing_labels, predictions_LSTM_model_3)
#print('Precision with early stopping and regularization: %f' % precision)

# Calculating recall for each models

# For CNN
# recall: tp / (tp + fnpredictions
recall = recall_score(testing_labels, predictions_DCmodel_1)
#print('Recall without regularization or dropout: %f' % recall)

recall = recall_score(testing_labels, predictions_DCmodel_2)
#print('Recall with early stopping and dropout: %f' % recall)

recall = recall_score(testing_labels, predictions_DCmodel_3)
#print('Recall with early stopping and regularization : %f' % recall)



# For LSTM
# recall: tp / (tp + fnpredictions
recall = recall_score(testing_labels, predictions_LSTM_model_1)
#print('Recall without regularization or dropout: %f' % recall)

recall = recall_score(testing_labels, predictions_LSTM_model_2)
#print('Recall with early stopping and dropout: %f' % recall)

recall = recall_score(testing_labels, predictions_LSTM_model_3)
#print('Recall with early stopping and regularization : %f' % recall)
# Calculating F1-score for each models

# For CNN and LSTM 

# f1: 2 tp / (2 tp + fp + fn)
f1_CNN = f1_score(testing_labels, predictions_DCmodel_1)
f1_LSTM = f1_score(testing_labels, predictions_LSTM_model_1)
print('Without regularization or dropout, F1 score of CNN = %f, and F1 score of LSTM = %f' % (f1_CNN, f1_LSTM))

f1_CNN = f1_score(testing_labels, predictions_DCmodel_2)
f1_LSTM = f1_score(testing_labels, predictions_LSTM_model_2)
print('With early stopping and dropout, F1 score of CNN = %f, and F1 score of LSTM = %f' % (f1_CNN, f1_LSTM))

f1_CNN = f1_score(testing_labels, predictions_DCmodel_3)
f1_LSTM = f1_score(testing_labels, predictions_LSTM_model_3)
print('With early stopping and regularization, F1 score of CNN = %f, and F1 score of LSTM = %f' % (f1_CNN, f1_LSTM))
test1 = tokenizer.texts_to_sequences(test.text)
test1 = pad_sequences(test1,maxlen = max_length, padding = padding_type,truncating = trunc_type)
predictions = LSTM_model_3.predict(test1).reshape(-1)
predictions = np.where(predictions<0.5,0,1)
my_submission= pd.DataFrame({'id': test.id,'target':predictions})
my_submission.to_csv('submission.csv',index=False)