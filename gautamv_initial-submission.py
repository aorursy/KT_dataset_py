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


import matplotlib.pyplot as plt



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import numpy



from keras.models import Model, Sequential

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, LSTM, SpatialDropout1D, Dropout, BatchNormalization

from keras.optimizers import RMSprop, Adam, Nadam, Adagrad

from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# fix random seed for reproducibility

numpy.random.seed(42)





from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import sklearn.metrics as metrics



from keras.optimizers import Adam, Adadelta, Nadam, RMSprop, SGD

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import plotly.express as px

import plotly.graph_objects as go
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.info()
train[train['keyword'].notnull()]['keyword'].unique()
train['keyword'].isnull().sum()
train['target'].value_counts(normalize=True)
train['keyword'] = train['keyword'].fillna("")
train['keyword'] = train['keyword'].str.replace("%20", " ")
train['keyword'].value_counts()[:10]
CONTRACTION_MAP = {

    "ain't": "is not",

    "aren't": "are not",

    "can't": "cannot",

    "can't've": "cannot have",

    "'cause": "because",

    "could've": "could have",

    "couldn't": "could not",

    "couldn't've": "could not have",

    "didn't": "did not",

    "doesn't": "does not",

    "don't": "do not",

    "hadn't": "had not",

    "hadn't've": "had not have",

    "hasn't": "has not",

    "haven't": "have not",

    "he'd": "he would",

    "he'd've": "he would have",

    "he'll": "he will",

    "he'll've": "he he will have",

    "he's": "he is",

    "how'd": "how did",

    "how'd'y": "how do you",

    "how'll": "how will",

    "how's": "how is",

    "I'd": "I would",

    "I'd've": "I would have",

    "I'll": "I will",

    "I'll've": "I will have",

    "I'm": "I am",

    "I've": "I have",

    "i'd": "i would",

    "i'd've": "i would have",

    "i'll": "i will",

    "i'll've": "i will have",

    "i'm": "i am",

    "i've": "i have",

    "isn't": "is not",

    "it'd": "it would",

    "it'd've": "it would have",

    "it'll": "it will",

    "it'll've": "it will have",

    "it's": "it is",

    "let's": "let us",

    "ma'am": "madam",

    "mayn't": "may not",

    "might've": "might have",

    "mightn't": "might not",

    "mightn't've": "might not have",

    "must've": "must have",

    "mustn't": "must not",

    "mustn't've": "must not have",

    "needn't": "need not",

    "needn't've": "need not have",

    "o'clock": "of the clock",

    "oughtn't": "ought not",

    "oughtn't've": "ought not have",

    "shan't": "shall not",

    "sha'n't": "shall not",

    "shan't've": "shall not have",

    "she'd": "she would",

    "she'd've": "she would have",

    "she'll": "she will",

    "she'll've": "she will have",

    "she's": "she is",

    "should've": "should have",

    "shouldn't": "should not",

    "shouldn't've": "should not have",

    "so've": "so have",

    "so's": "so as",

    "that'd": "that would",

    "that'd've": "that would have",

    "that's": "that is",

    "there'd": "there would",

    "there'd've": "there would have",

    "there's": "there is",

    "they'd": "they would",

    "they'd've": "they would have",

    "they'll": "they will",

    "they'll've": "they will have",

    "they're": "they are",

    "they've": "they have",

    "to've": "to have",

    "wasn't": "was not",

    "we'd": "we would",

    "we'd've": "we would have",

    "we'll": "we will",

    "we'll've": "we will have",

    "we're": "we are",

    "we've": "we have",

    "weren't": "were not",

    "what'll": "what will",

    "what'll've": "what will have",

    "what're": "what are",

    "what's": "what is",

    "what've": "what have",

    "when's": "when is",

    "when've": "when have",

    "where'd": "where did",

    "where's": "where is",

    "where've": "where have",

    "who'll": "who will",

    "who'll've": "who will have",

    "who's": "who is",

    "who've": "who have",

    "why's": "why is",

    "why've": "why have",

    "will've": "will have",

    "won't": "will not",

    "won't've": "will not have",

    "would've": "would have",

    "wouldn't": "would not",

    "wouldn't've": "would not have",

    "y'all": "you all",

    "y'all'd": "you all would",

    "y'all'd've": "you all would have",

    "y'all're": "you all are",

    "y'all've": "you all have",

    "you'd": "you would",

    "you'd've": "you would have",

    "you'll": "you will",

    "you'll've": "you will have",

    "you're": "you are",

    "you've": "you have",

    "&amp;": '&',

    "&gt;": '>',

    "&lt;": '<',

    "<3": "love",

    "2way ": "two way ",

    " 1-0-1 ": " one on one ",

    " 1-O-1 ": " one on one ",

    " 1-o-1 ": " one on one",

    " its ": " it is ",

    "verygood": "very good",

    "verybad": "very bad",

    "what's": "what is",

    "that's": "that is",

    "thats": "that is",

    "'m": " am ",

    "'s": " is ",

    "'ve": " have ",

    "wasn't": "was not",

    "hasn't": "has not",

    "haven't": "have not",

    "shalln't": "shall not",

    "hadn't": "had not",

    "won't": "will not",

    "isn't": "is not",

    "don't": "do not",

    " isnt ": " is not ",

    "wont": "will not",

    "hadnt": "had not",

    "hasnt": "has not",

    "dont": "do not",

    "`m": " am ",

    " im ": " i am ",

    "i'm": " i am ",

    "'re": " are ",

    "'d": " would ",

    "'ll": " will ",

    "'scuse": " excuse ",

    "didnt": "did not",

    "didnot": "did not",

    "doesnot": "does not",

    " wil ": " will ",

    " iam ": " i am ",

    " dnt ": " do not ",

    "rightnow": "right now",

    "havenot": "have not",

    "havent": "have not",

    "doesnt": "does not",

    "donot": "do not",

    " alot ": " a lot ",

    " whats ": " what is ",

    "whats": "what is",

    "wasnt": "was not",

    "gonna": "going to",

    "wanna": "want to",

    "worklife": "work life",

    " aswell ": " as well ",

    "couldnt": "could not",

    "shouldnt": "should not",

    "wouldnt": "would not",

    " theres ": "there is ",

    " arent ": " are not ",

    "followup": "follow up",

    " ive ": " i have ",

    "eventhough": "even though",

    "kinda": "kind of",

    "youre": "you are",

    "thanku": "thank you",

    " n ": " and ",

    " k ": " ok",

    "'s": " ",

    ":(": "sad_smiley",

    ":-(": "sad_smiley",

    ":-)": "happy_smiley",

    ":)": "happy_smiley",

    "-)": "happysmiley",

    ":p": "sarcasticsmiley",

    ":P": "sarcasticsmiley",

    ":d": "delightedsmiley",

    " hv ": " have ",

    " 2way ": " two way ",

    " pls ": " please ",

    " plz ": " please ",

    " yea ": " yeah ",

    " im ": " i am ",

    " iam ": " i am ",

    " km ": " kilometer ",

    " hr ": " hour ",

    " dnt ": " do not ",

    " mgt ": " management ",

    " lyk ": " like ",

    " ntg ": " nothing ",

    " gud ": " good ",

    " yrs ": " years ",

    " yaa ": " yeah ",

    " tq ": " thank you ",

    " idk ": " i do not know ",

    " wat ": " what ",

    'hwy': "highway"

}
import re



def split_camel_case_words(x):

    if '#' in x:

        return re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', x))

    return x



def extract_text_between_quotes(x):    

    if len(re.findall('\'(.*?)\'', x)) > 1:

        return x

    extracted = " ".join(re.findall('\'(.*?)\'', x))

    if len(extracted.split(" ")) > 6:

        return extracted

    return x
#     x = re.sub(r'(?:[\£\$\€]{1}[,\d]+.?\d*-[\£\$\€]{1}[,\d]+.?\d*)', ' PRICE_RANGE_TAG ', x) # price range tag

#     x = re.sub(r'(?:[\£\$\€]{1}[,\d]+[^ -]?\d*)', ' PRICE_TAG ', x) # Keep Price tags which involves currency

#     x = re.sub(r'([\d]{1,4}-[\d]{1,2}-[\d]{2,4})', ' DATE_TAG ', x)

#     x = re.sub(r'((%s) [\d]{1,2}, [\d]{4})'%(months_re), ' DATE_TAG ', x)

#     x = re.sub(r'(?:[\d]+[+]? (years|year|months|month|weeks|week|days|day))', ' TIME_PERIOD_TAG ', x)

#     x = re.sub(r'([\d]{1,2}:[\d]{1,2}[ ]?(pm|p.m.|am|a.m.)?)', ' TIME_TAG ', x) # TIME_TAG

#     x = re.sub(r'([\d] (pm|p.m.|am|a.m.))', ' TIME_TAG ', x) # Time Tag

#     x = re.sub(r'([\d]{1,2}(p.m.|p.m|pm|a.m.|a.m|am))', ' TIME_TAG ', x)

   

#     # keep TIME_TAGS for detecting literal time strings

#     x = re.sub(r'(?:[\d]/[\d] (hours|hour|minutes|minute|mins|min|seconds|second|sec))', ' TIME_PERIOD_TAG ', x)

#     x = re.sub(r'(?:[\d]+[+]? (hours|hour|minutes|minute|mins|min|seconds|second|sec))', ' TIME_PERIOD_TAG ', x)

   

#     x = re.sub(r'(?:[\d]+[\']s)', ' TIME_PERIOD_THEME_TAG ', x) # Time period tag

#     x = re.sub(r'([\d]+%)', ' DISCOUNT_TAG ', x) # keep discount where numbers appears with percentages

#     x = re.sub(r'([\d]+[ ]?(times|time|x))', ' REPETITION_TAG ', x)

#     x = re.sub(r'([\d]+,[\d]+)', 'NUMBER_TAG', x)

# #     x = x.replace("'s", "")

# #     x = x.replace("'", " ' ")

#     x = re.sub(r'([\d]+(st|nd|rd|th))', ' TIMETH_TAG ', x)

#     x = re.sub(r'([\d]/[\d])', ' PORTION_TAG ', x)

#     x = re.sub(r'([\d]-[\d])', ' RANGE_TAG ', x)

#     x = re.sub('\d+', ' NUMBER_TAG ', x) # Replace numbers with tags
import nltk

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import re

import itertools



months_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',

               'sep', 'oct', 'nov', 'dec', 'january', 'february', 'march',

               'april', 'may', 'june', 'july', 'august',

               'september', 'october', 'november', 'december']

months_re = "|".join(months_list)





def contains_non_ascii(s):

    return any(ord(i)>127 for i in s)



def preprocess_text(x):

    if sum([ord(i) > 127 for i in x]) > 30:

        return None

    

    x = re.sub(r'http\S+', " ", x) # Remove URLs

    x = re.sub(r'ww\S+', ' ', x)

    x = re.sub(r"\x89Û_", "", x)

    x = re.sub(r"\x89ÛÒ", "", x)

    x = re.sub(r"\x89ÛÓ", "", x)

    x = re.sub(r"\x89ÛÏWhen", "When", x)

    x = re.sub(r"\x89ÛÏ", "", x)

    x = re.sub(r"China\x89Ûªs", "China's", x)

    x = re.sub(r"let\x89Ûªs", "let's", x)

    x = re.sub(r"\x89Û÷", "", x)

    x = re.sub(r"\x89Ûª", "", x)

    x = re.sub(r"\x89Û\x9d", "", x)

    x = re.sub(r"å_", "", x)

    x = re.sub(r"\x89Û¢", "", x)

    x = re.sub(r"\x89Û¢åÊ", "", x)

    x = re.sub(r'\x89ã¢', "", x)

    x = re.sub(r"fromåÊwounds", "from wounds", x)

    x = re.sub(r"åÊ", "", x)

    x = re.sub(r"åÈ", "", x)

    x = re.sub(r"JapÌ_n", "Japan", x)    

    x = re.sub(r"Ì©", "e", x)

    x = re.sub(r"å¨", "", x)

    x = re.sub(r"SuruÌ¤", "Suruc", x)

    x = re.sub(r"åÇ", "", x)

    x = re.sub(r"å£3million", "3 million", x)

    x = re.sub(r"åÀ", "", x)

    x = re.sub(r"don\x89Ûªt", "do not", x)

    x = re.sub(r"I\x89Ûªm", "I am", x)

    x = re.sub(r"you\x89Ûªve", "you have", x)

    x = re.sub(r"it\x89Ûªs", "it is", x)

    x = re.sub(r"%20", " ", x)

    x = re.sub(r'[\r\n\t]', " ", x)

    x = re.sub(r'w/', " with ", x)

    x = re.sub(r'P/U', " pickup ", x)

    x = re.sub(r'xb1', "xbox", x)

    x = re.sub(r' ps ', " playstation ", x)

    x = re.sub(r'cya', "see you", x)

    x = re.sub(r"lmao", "laughing my ass off", x)

    x = re.sub(r"w/e", "whatever", x)

    x = re.sub(r"USAgov", "USA government", x)

    x = re.sub(r"recentlu", "recently", x)

    x = re.sub(r"Ph0tos", "Photos", x)

    x = re.sub(r"amirite", "am I right", x)

    x = re.sub(r"exp0sed", "exposed", x)

    x = re.sub(r"<3", "love", x)

    x = re.sub(r"amageddon", "armageddon", x)

    x = re.sub(r"Trfc", "Traffic", x)

    x = re.sub(r"rea\x89Û_", "real", x)

    

    x = " ".join([i for i in x.split(" ") if not (i.startswith("@") or i.startswith(".@"))])

    x = " ".join([split_camel_case_words(i) for i in x.split(" ")])

    

    x = x.lower()

    for s, rep in CONTRACTION_MAP.items():

        x = x.replace(s, rep)

    x = extract_text_between_quotes(x)

    x = " ".join([word for word in x.split() if not contains_non_ascii(word)])

    

    #x = re.sub(r'[\"\']', "", x)

    x = re.sub("[\(\[].*?[\)\]]", "", x) # Remove text between brackets

    #x = re.sub( r'[\\*=~_|<>{}*]', ' ', x)

    x = re.sub('\d+', ' ', x) # Replace numbers with tags

    # Standardizing words: Removing more than 2 repeating characters in the text

    x = ''.join(''.join(s)[:2] for _, s in itertools.groupby(x))

    

    punctuations = '@#+&*[]-%:/();$=><|{}^' + "'`"

    for p in punctuations:

        x = x.replace(p, ' ')

    for p in ',.?!':

        x = x.replace(p, " "+p+" ")

    x = " ".join([lemmatizer.lemmatize(i) if i not in stop_words else i for i in x.split(" ")])

    

    x = x.replace('...', ' ... ')

    if '...' not in x:

        x = x.replace('..', ' ... ')

   

    x = re.sub(r'\s+', ' ', x) # Remove Extra spaces

    x = x.strip()

    return x
train['text_cleaned'] = train['text'].apply(preprocess_text)
train['text'].values[100:150]
train['text_cleaned'].values[100:150]
# train['text_cleaned'] = train['keyword'] + " " + train['text_cleaned']
train['norm_len'] = train['text_cleaned'].apply(lambda x: len(x.split(" ")))
train['norm_len'].hist()
train['norm_len'].min()
train = train[train['norm_len'] >2]
train['norm_len'].quantile([0.95, 0.975, 0.99])
from collections import defaultdict

word_dict = defaultdict(int)

for seq in train['text_cleaned'].values:

    for w in seq.split(" "):

        word_dict[w] += 1
len(word_dict)
for i in range(0, 10, 1):

    print("Freq greater than %d has %s words in vocabulary" %(i, len([k for k in word_dict.values() if k > i])))
word_dict_df = pd.DataFrame({

    'word': list(word_dict.keys()),

    'freq': list(word_dict.values())

})
train[train['text_cleaned'].str.contains("collegeradi")]['text_cleaned'].values[0]
word_dict_df[word_dict_df['freq'] == 1]
X_train, X_test, y_train, y_test = train_test_split(train['text_cleaned'], train['target'].values, test_size=0.2,

                                                    random_state=42, stratify=train['target'].values)
for i in range(0, 10, 1):

    print("Freq greater than %d has %s words in vocabulary" %(i, len([k for k in word_dict.values() if k > i])))
vocab_size = len(word_dict) + 1

embedding_dim = 64

max_length = 26

oov_tok = '<OOV>'
tokenizer = Tokenizer(num_words=vocab_size, filters="", oov_token=oov_tok)

tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index

dict(list(word_index.items())[0:10])
train_sequences = tokenizer.texts_to_sequences(X_train)

sequences_matrix = sequence.pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
valid_sequences = tokenizer.texts_to_sequences(X_test)

valid_seq_matrix = sequence.pad_sequences(valid_sequences, maxlen=max_length, padding='post', truncating='post')
from keras.layers import Bidirectional
from keras import backend as K



def get_recall(y_true, y_pred):

    """Recall metric.



    Only computes a batch-wise average of recall.



    Computes the recall, a metric for multi-label classification of

    how many relevant items are selected.

    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def get_precision(y_true, y_pred):

    """Precision metric.



    Only computes a batch-wise average of precision.



    Computes the precision, a metric for multi-label classification of

    how many selected items are relevant.

    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision

    

def f1(y_true, y_pred):    

    precision = get_precision(y_true, y_pred)

    recall = get_recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def RNN(max_length, vocab_size, embedding_dim):

    model = Sequential()

    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))

    model.add(SpatialDropout1D(0.5))

    model.add(Bidirectional(LSTM(max_length, dropout=0.5, recurrent_dropout=0.5)))

    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    

    return model
model = RNN(max_length, vocab_size, embedding_dim)
model.summary()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, 

                              mode='auto', min_delta=0.005, cooldown=0, min_lr=0)

es = EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
optimizer = Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
#optimizer = SGD(learning_rate=0.0001, momentum=0.9, nesterov=True, name="SGD")

#optimizer=Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)
model.compile(loss='binary_crossentropy', 

              optimizer=optimizer,

              metrics=[f1])
history = model.fit(sequences_matrix, y_train, batch_size=64, epochs=30,

                    validation_data=(valid_seq_matrix, y_test), callbacks=[reduce_lr, es])
fig = go.Figure()

fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))

fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))            

fig.update_layout(title="Loss per Each epochs")

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(y=history.history['f1'], name='Training F1-score'))

fig.add_trace(go.Scatter(y=history.history['val_f1'], name='Validation F1-score'))            

fig.update_layout(title=" Accuracy per each epoch")

fig.show()
y_pred = model.predict(valid_seq_matrix)
y_pred = y_pred.ravel()
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

roc_auc = metrics.auc(fpr, tpr)


plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
metrics.f1_score(y_test, np.where(y_pred >= 0.5, 1, 0))
test['text_cleaned'] = test['text'].apply(preprocess_text)
test_sequences = tokenizer.texts_to_sequences(test['text_cleaned'])

test_seq_matrix = sequence.pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')
test_pred = model.predict(test_seq_matrix)
test_pred = test_pred.ravel()
test['target'] = test_pred
test['target'] = np.where(test['target'] >= 0.5, 1, 0)
test.head()
test[['id', 'target']].to_csv('lstm_submission2.csv', index=None)