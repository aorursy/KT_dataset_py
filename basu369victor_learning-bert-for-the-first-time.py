# We will use the official tokenization script created by the Google team

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import numpy as np

import pandas as pd

import re

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub

#from nltk.corpus import stopwords

#from nltk.stem.porter import PorterStemmer



import tokenization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
BASE_PATH = "/kaggle/input/nlp-getting-started/"
train =pd.read_csv(BASE_PATH + "train.csv")

train.head()
test =pd.read_csv(BASE_PATH + "test.csv")

test.head()
%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
text = "This is a Goat, and I am riding a Boat...."

tokenize_ = tokenizer.tokenize(text)

print("Text after tokenization: ")

print(tokenize_)

max_len = 25



text = tokenize_[:max_len-2]

input_sequence = ["[CLS]"] + text + ["[SEP]"]

pad_len = max_len - len(input_sequence)



print("After adding [CLS] and [SEP]: ")

print(input_sequence)

tokens = tokenizer.convert_tokens_to_ids(input_sequence)

print("After converting Tokens to Id: ")

print(tokens)

tokens += [0] * pad_len

print("tokens: ")

print(tokens)

pad_masks = [1] * len(input_sequence) + [0] * pad_len

print("Pad Masking: ")

print(pad_masks)

segment_ids = [0] * max_len

print("Segment Ids: ")

print(segment_ids)
def pre_Process_data(documents, tokenizer, max_len=512):

    '''

    For preprocessing we have regularized, transformed each upper case into lower case, tokenized,

    Normalized and remove stopwords. For normalization, we have used PorterStemmer. Porter stemmer transforms 

    a sentence from this "love loving loved" to this "love love love"

    

    '''

    all_tokens = []

    all_masks = []

    all_segments = []

    print("Pre-Processing the Data.........\n")

    for data in documents:

        review = re.sub('[^a-zA-Z]', ' ', data)

        url = re.compile(r'https?://\S+|www\.\S+')

        review = url.sub(r'',review)

        html=re.compile(r'<.*?>')

        review = html.sub(r'',review)

        emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

        review = emoji_pattern.sub(r'',review)

        text = tokenizer.tokenize(review)

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
input_word_id = Input(shape=(max_len,),dtype=tf.int32, name="input_word_ids")

input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

segment_id = Input(shape=(max_len,), dtype=tf.int32, name = "segment_id")



_, sequence_output = bert_layer([input_word_id, input_mask, segment_id])

clf_output = sequence_output[:, 0, :]

model = Model(inputs=[input_word_id, input_mask, segment_id],outputs=clf_output)

model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

print("shape of _ layer of BERT: "+str(_.shape))

print("shape of last layer of BERT: "+str(sequence_output.shape))
def build_model(bert_layer, max_len=512):

    input_word_id = Input(shape=(max_len,),dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_id = Input(shape=(max_len,), dtype=tf.int32, name = "segment_id")

    

    _, sequence_output = bert_layer([input_word_id, input_mask, segment_id])

    clf_output = sequence_output[:, 0, :]

    dense_layer1 = Dense(units=256,activation='relu')(clf_output)

    dense_layer1 = Dropout(0.4)(dense_layer1)

    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)

    dense_layer2 = Dropout(0.4)(dense_layer2)

    out = Dense(1, activation='sigmoid')(dense_layer2)

    

    model = Model(inputs=[input_word_id, input_mask, segment_id],outputs=out)

    model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
train_input = pre_Process_data(train.text.values, tokenizer, max_len=260)

test_input = pre_Process_data(test.text.values, tokenizer, max_len=260)

train_labels = train.target.values
model = build_model(bert_layer, max_len=260)

model.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

train_history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=10,

    callbacks=[checkpoint],

    batch_size=32

)
submission = pd.read_csv(BASE_PATH + "sample_submission.csv")

submission.head()
model.load_weights('model.h5')

test_pred = model.predict(test_input)



submission['target'] = test_pred.round().astype(int)

submission.to_csv('submission.csv', index=False)


