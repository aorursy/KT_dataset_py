import re

import string

import operator

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

%matplotlib inline



from keras import backend as K

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, Bidirectional, Lambda, Reshape,GlobalMaxPool1D

from keras.optimizers import Adam

from keras.models import Model, Sequential

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers



from sklearn.model_selection import train_test_split

from sklearn import metrics



import nltk

from nltk.corpus import stopwords







import os

print(os.listdir("../input"))

df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")







print("Train shape : ",df.shape)

print("Test shape : ",test_df.shape)

print(df.isnull().sum())

print(test_df.isnull().sum())
df['target'].value_counts().plot(kind = 'pie', labels = ['Sinceras', 'Toxicas '],autopct='%1.0f%%')


import re

cList = {

  "ain't": "am not",

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

  "he'll've": "he will have",

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

  "isn't": "is not",

  "it'd": "it had",

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

  "so's": "so is",

  "that'd": "that would",

  "that'd've": "that would have",

  "that's": "that is",

  "there'd": "there had",

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

  "we'd": "we had",

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

  "y'alls": "you alls",

  "y'all'd": "you all would",

  "y'all'd've": "you all would have",

  "y'all're": "you all are",

  "y'all've": "you all have",

  "you'd": "you had",

  "you'd've": "you would have",

  "you'll": "you you will",

  "you'll've": "you you will have",

  "you're": "you are",

  "you've": "you have"

}



c_re = re.compile('(%s)' % '|'.join(cList.keys()))



punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

mispell_dict = {'advanatges': 'advantages', 'irrationaol': 'irrational' , 'defferences': 'differences','lamboghini':'lamborghini','hypothical':'hypothetical', 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon'}
def expandContractions(text, c_re=c_re):

    def replace(match):

        return cList[match.group(0)]

    return c_re.sub(replace, text)





def clean_numbers(x):

    x = re.sub('[0-9]{5,}', ' number ', x)

    x = re.sub('[0-9]{4}', ' number ', x)

    x = re.sub('[0-9]{3}', ' number ', x)

    x = re.sub('[0-9]{2}', ' number ', x)

    return x







def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text









def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x
df['question_text'] = df['question_text'].apply(lambda x: expandContractions(x))

df['question_text'] = df['question_text'].apply(lambda x: clean_numbers(x))

df['question_text'] = df['question_text'].apply(lambda x: x.lower())

df['question_text'] = df['question_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

df['question_text'] = df['question_text'].apply(lambda x: correct_spelling(x, mispell_dict))

df.head(100)

test_df['question_text'] = test_df['question_text'].apply(lambda x: expandContractions(x))

test_df['question_text'] = test_df['question_text'].apply(lambda x: clean_numbers(x))

test_df['question_text'] = test_df['question_text'].apply(lambda x: x.lower())

test_df['question_text'] = test_df['question_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

test_df['question_text'] = test_df['question_text'].apply(lambda x: correct_spelling(x, mispell_dict))

test_df.head()
df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
# some config values 

embed_size = 200 # how big is each word vector

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 100 # max number of words in a question to use



# fill up the missing values

x_train = df_train["question_text"].fillna("_na_").values

x_val = df_val["question_text"].fillna("_na_").values



# Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(x_train))

x_train = tokenizer.texts_to_sequences(x_train)

x_val = tokenizer.texts_to_sequences(x_val)



# Pad the sentences 

x_train = pad_sequences(x_train, maxlen=maxlen)

x_val = pad_sequences(x_val, maxlen=maxlen)



# Get the target values

y_train = df_train['target'].values

y_val = df_val['target'].values
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

def get_coefs(word,*arr): 

    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: 

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: 

        embedding_matrix[i] = embedding_vector
model = Sequential()

model.add(Embedding(max_features, 

                    embed_size, 

                    weights=[embedding_matrix]))

model.add(Bidirectional(CuDNNGRU(64, return_sequences=True)))

model.add(Bidirectional(CuDNNGRU(32, return_sequences=True)))

model.add(GlobalMaxPool1D())

model.add(Dropout(0.1))

model.add(Dense(16, activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train, batch_size=512, epochs=2, validation_data=(x_val, y_val))


x_test = test_df["question_text"].fillna("_na_").values



x_test = tokenizer.texts_to_sequences(x_test)



x_test = pad_sequences(x_test, maxlen=maxlen)
y_test = model.predict([x_test], batch_size=1024, verbose=1)

y_test = (y_test > 0.4).astype(int)

test_df = pd.DataFrame({"qid": test_df["qid"].values})

test_df['prediction'] = y_test

test_df.to_csv("submission.csv", index=False)


test_df['prediction'].value_counts().plot(kind = 'pie', labels = ['Sinceras', 'Toxicas '], autopct='%1.0f%%')