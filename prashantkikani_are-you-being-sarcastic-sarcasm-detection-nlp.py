# pandas to open data files & processing it.

import pandas as pd

# to see all columns

pd.set_option('display.max_columns', None)

# To see whole text

pd.set_option('max_colwidth', -1)



# numpy for numeric data processing

import numpy as np



# keras for deep learning model creation

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras.utils import plot_model



# to fix random seeds

import random

import tensorflow as tf

import torch

import os



# Regular Expression for text cleaning

import re



# to track the progress - progress bar

from tqdm.notebook import tqdm
sarcasm_data = pd.read_csv("../input/sarcasm/train-balanced-sarcasm.csv")

print(sarcasm_data.shape)

sarcasm_data.head()
sarcasm_data.drop(['author', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc', 'parent_comment'], axis=1, inplace=True)

# remove empty rows

sarcasm_data.dropna(inplace=True)

sarcasm_data.head()
sarcasm_data['label'].value_counts()
mispell_dict = {"ain't": "is not", "cannot": "can not", "aren't": "are not", "can't": "can not", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",

                "doesn't": "does not",

                "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",

                "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",

                "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",

                "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",

                "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",

                "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",

                "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would",

                "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",

                "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",

                "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",

                "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "wont": "will not", "won't've": "will not have", "would've": "would have",

                "wouldn't": "would not",

                "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",

                "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color',

                'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',

                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',

                'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I',

                'theBest': 'the best', 'howdoes': 'how does', 'Etherium': 'Ethereum',

                'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what',

                'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}



mispell_dict = {k.lower(): v.lower() for k, v in mispell_dict.items()}
def preprocessing_text(s):

    # making our string lowercase & removing extra spaces

    s = str(s).lower().strip()

    

    # remove contractions.

    s = " ".join([mispell_dict[word] if word in mispell_dict.keys() else word for word in s.split()])

    

    # removing \n

    s = re.sub('\n', '', s)

    

    # put spaces before & after punctuations to make words seprate. Like "king?" to "king", "?".

    s = re.sub(r"([?!,+=—&%\'\";:¿।।।|\(\){}\[\]//])", r" \1 ", s)

    

    # Remove more than 2 continues spaces with 1 space.

    s = re.sub('[ ]{2,}', ' ', s).strip()

    

    return s
# apply preprocessing_text function

sarcasm_data['comment'] = sarcasm_data['comment'].apply(preprocessing_text)

sarcasm_data.head()
# total unique words we are going to use.

TOTAL_WORDS = 40000



# max number of words one sentence can have

MAX_LEN = 50



# width of of 1D embedding vector

EMBEDDING_SIZE = 300
%%time

tokenizer = Tokenizer(num_words=TOTAL_WORDS)

tokenizer.fit_on_texts(list(sarcasm_data['comment']))



train_data = tokenizer.texts_to_sequences(sarcasm_data['comment'])

train_data = pad_sequences(train_data, maxlen = MAX_LEN)

target = sarcasm_data['label']
%%time

EMBEDDING_FILE = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'



def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')



embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in tqdm(open(EMBEDDING_FILE)))



word_index = tokenizer.word_index

nb_words = min(TOTAL_WORDS, len(word_index))

embedding_matrix = np.zeros((nb_words, EMBEDDING_SIZE))
for word, i in tqdm(word_index.items()):

    if i >= TOTAL_WORDS: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
embedding_matrix.shape
def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    tf.random.set_seed(seed)



# We fix all the random seed so that, we can reproduce the results.

seed_everything(2020)
input_layer = Input(shape=(MAX_LEN,))



embedding_layer = Embedding(TOTAL_WORDS, EMBEDDING_SIZE, weights = [embedding_matrix])(input_layer)



LSTM_layer = Bidirectional(LSTM(128, return_sequences = True))(embedding_layer)

maxpool_layer = GlobalMaxPool1D()(LSTM_layer)



dense_layer_1 = Dense(64, activation="relu")(maxpool_layer)

dropout_1 = Dropout(0.5)(dense_layer_1)



dense_layer_2 = Dense(32, activation="relu")(dropout_1)

dropout_2 = Dropout(0.5)(dense_layer_2)



output_layer = Dense(1, activation="sigmoid")(dropout_2)



model = Model(input=input_layer, output=output_layer)



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
plot_model(model, show_shapes=True)
BATCH_SIZE = 512

EPOCHS = 2



history = model.fit(

    train_data, target,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    # We are using randomly selected 20% sentences as validation data.

    validation_split=0.2

)
sarcasm_data[sarcasm_data['label']==1].sample(20)
sentence = "sun rises from the east"

sentence = preprocessing_text(sentence)

print(sentence)



sentence = tokenizer.texts_to_sequences([sentence])

sentence = pad_sequences(sentence, maxlen = MAX_LEN)

sentence
# Make the prediction.

prediction = model.predict(sentence)

prediction[0][0]
print("So, it's saying sentence have probability of %.3f percent"%(prediction[0][0]*100))
sentence = "Isn't it great that, your girlfriend dumped you?"

sentence = preprocessing_text(sentence)

print(sentence)



sentence = tokenizer.texts_to_sequences([sentence])

sentence = pad_sequences(sentence, maxlen = MAX_LEN)

sentence
# Make the prediction.

prediction = model.predict(sentence)

prediction[0][0]
print("So, it's saying sentence have probability of %.3f percent"%(prediction[0][0]*100))