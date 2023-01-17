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
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

train.head()
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

test.head()
full = train.append(test)

full.shape
full = full.reset_index(drop=True)

full["train"] = 0

full.loc[train.index,"train"] = 1

full.head(10)
full = full.loc[(full.text.notna())] 

full.text = full.text.str.strip()
full["Has_BW"] = full.text.str.contains("\*\*\*").astype(int)

full.head()
full["clean_text"] = full["text"]

full.head()
import re

def clean_text(text):

    lower = text.lower()

    no_links = re.sub("((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*","",lower)

    no_alpha_num = re.sub("[^a-zA-Z\s\*]","", no_links)

    no_bw = no_alpha_num.replace("**","*n").replace("*n*n","*n")

    no_extra_spaces =no_bw.replace("  "," ")

    return no_bw

    
full.clean_text = full.clean_text.apply(clean_text)

full.head()
full.text = full.text.str.replace("((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*","")

full.head()
train = full.loc[full.train == 1]

test = full.loc[full.train == 0]
train["WRONG_DATA"] = train.apply(lambda x: x["selected_text"] not in x["text"],axis=1).astype(int)

train.loc[train.clean_text == '','WRONG_DATA'] = 1

train = train.loc[train.WRONG_DATA == 0]

train.head()
train["selected_clean_text"] = train.selected_text.apply(clean_text)

train.head()
train = train.loc[train.selected_clean_text != ""]

train.shape
max_len = full.text.str.split().apply(len).max()

max_len
def convert_text(texts):

    bio_texts_df = pd.DataFrame([])

    for text in texts:

        text_id_sentiment_data = list()

        selected_text = text['selected_clean_text']

        clean_text = text['clean_text']

        split_text = clean_text.split(selected_text)

        for t in [split_text[0],selected_text,split_text[1]]:

            if t != '':

                for w in t.split():

                    text_id_sentiment_data.append({

                            'word':w,

                            'target':1 if t == selected_text else 0,

                            'textID': text['textID']

                        })

        bio_texts_df = bio_texts_df.append(pd.DataFrame(text_id_sentiment_data))

    return bio_texts_df.reset_index(drop=True)
data_converted_train = convert_text(train.to_dict(orient='records'))
train_text_ids = data_converted_train.textID.drop_duplicates().sample(frac=0.9)

test_text_ids = data_converted_train.loc[~(data_converted_train.textID.isin(train_text_ids))].textID.drop_duplicates()
data_converted_train_full = data_converted_train.merge(train[["textID","sentiment"]])

data_converted_train_full.head()
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import numpy as np 

np.random.seed(777)

def create_sentiment_data(data,sentiment):

    sentiment_data = data.loc[data.sentiment == sentiment]

    train_text_ids = sentiment_data.textID.drop_duplicates().sample(frac=0.9).sort_values()

    test_text_ids = sentiment_data.loc[~(sentiment_data.textID.isin(train_text_ids))].textID.drop_duplicates().sort_values()

    agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),s["target"].values.tolist())]

    train_sentiment_data = sentiment_data.loc[sentiment_data.textID.isin(train_text_ids)].sort_values(by='textID')

    test_sentiment_data = sentiment_data.loc[sentiment_data.textID.isin(test_text_ids)].sort_values(by='textID')

    

    train_grouped_sentences = train_sentiment_data.groupby("textID").apply(agg_func)

    train_sentences = [s for s in train_grouped_sentences]

    train_raw_sentences = [" ".join([w[0] for w in s]) for s in train_sentences]

    

    test_grouped_sentences = test_sentiment_data.groupby("textID").apply(agg_func)

    test_sentences = [s for s in test_grouped_sentences]

    test_raw_sentences = [" ".join([w[0] for w in s]) for s in test_sentences]

    

    vocab_length = 10000

    entity_text_tokenizer = Tokenizer(vocab_length)

    entity_text_tokenizer.fit_on_texts(train_raw_sentences)

    train_embedded_sentences = entity_text_tokenizer.texts_to_sequences(train_raw_sentences)

    train_padded_sentences = pad_sequences(train_embedded_sentences, max_len, padding='post')  



    test_embedded_sentences = entity_text_tokenizer.texts_to_sequences(test_raw_sentences)

    test_padded_sentences = pad_sequences(test_embedded_sentences, max_len, padding='post')

    

    

    train_tags = train_sentiment_data.target.unique()

    n_tags = len(train_tags)

    tags2index = {t:i for i,t in enumerate(train_tags)}

    train_target_tags = [[tags2index[w[1]] for w in s] for s in train_sentences]

    train_target_tags = pad_sequences(maxlen=max_len, sequences=train_target_tags, padding="post", value=0)

    

    test_tags = test_sentiment_data.target.unique()

    n_tags = len(test_tags)

    tags2index = {t:i for i,t in enumerate(test_tags)}

    test_target_tags = [[tags2index[w[1]] for w in s] for s in test_sentences]

    test_target_tags = pad_sequences(maxlen=max_len, sequences=test_target_tags, padding="post", value=0)

    

    return entity_text_tokenizer,train_padded_sentences,test_padded_sentences,train_target_tags,test_target_tags,train_text_ids,test_text_ids
positive_tokenizer,train_positive_sentences,test_positive_sentences,train_positive_tags,test_positive_tags,train_pos_ids,test_pos_ids = create_sentiment_data(data_converted_train_full,"positive")

train_text_ids.iloc[:5]
import tensorflow.keras as keras

from tensorflow.keras.layers import Concatenate

from tensorflow.keras import Model, Input

from tensorflow.keras.backend import clear_session

import gc

import numpy as np

from sklearn.preprocessing import LabelEncoder

import sklearn.metrics as sklm

from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda,Attention,GlobalAveragePooling1D,Conv1D,GlobalMaxPooling1D
vocab_length = 10000

n_tags=2
input_text = Input(shape=(max_len,))

embedding  = Embedding(vocab_length, 32, input_length=max_len)(input_text)

lstm_1 = Bidirectional(LSTM(units=32, return_sequences=True,

                       recurrent_dropout=0.2, dropout=0.2))(embedding)

lstm_2 = Bidirectional(LSTM(units=32, return_sequences=True,

                           recurrent_dropout=0.2, dropout=0.2))(lstm_1)

merge = Concatenate()([lstm_1,lstm_2])

output = TimeDistributed(Dense(n_tags, activation="softmax"))(merge)

model = Model(input_text, output)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()
import numpy as np
history = model.fit(train_positive_sentences, train_positive_tags, epochs=5, verbose=1,batch_size=64)
!pip install seqeval
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

test_pred = model.predict(np.array(test_positive_sentences))
def pred2label(pred):

    out = []

    for pred_i in pred:

        out_i = []

        for p in pred_i:

            p_i = np.argmax(p)

            out_i.append(str(p_i))

        out.append(out_i)

    return out

def test2label(pred):

    out = []

    for pred_i in pred:

        out_i = []

        for p in pred_i:

            out_i.append(str(p))

        out.append(out_i)

    return out

    

pred_labels = pred2label(test_pred)

test_labels = test2label(test_positive_tags)
pred_labels[:5]
test_labels[:5]
print(classification_report(test_labels, pred_labels))
test_texts = positive_tokenizer.sequences_to_texts(test_positive_sentences)
test_texts[:5]
def pred_to_selected_text(texts,preds):

    selected_texts = list()

    for index,text in enumerate(texts):

        pred_selected_text = preds[index]

        words = text.split()

        prediction = " ".join([words[i] for i,label in enumerate(pred_selected_text) if label == 1 and i < len(words)])

        selected_texts.append(prediction)

    return selected_texts
test_pos_ids.head()
train.loc[train.textID == "9339ee8e0b"]
train.text.values
pred_to_selected_text(test_texts,test_positive_tags)[:5]
y_test