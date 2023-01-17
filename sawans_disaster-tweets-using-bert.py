# We will use the official tokenization script created by the Google team

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

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

        
def build_model(bert_layer, max_len=512):

    

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(clf_output)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model

    
## Load bert from the tensorflow Hub



module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
## Load CSV files containing data



train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
## Importing required libaries for preprocssing



import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

import re 

nltk.download('punkt')

from nltk.tokenize import word_tokenize
## Define regular expressions, stopwords and lemmatizer



replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')

bad_symbols_re = re.compile('[^0-9a-z ]')

#links_re = re.compile('(www|http)\S+')

links_re = re.compile(r'http\S+')



Stopwords = set(stopwords.words('english'))

Stopwords.remove('no')

Stopwords.remove('not')



lemmatizer = nltk.stem.WordNetLemmatizer()
## Function to clean and prepare text



def text_prepare(text):

    """

        text: a string

        

        return: modified initial string

    """

    

    text = text.lower()  # lowercase text

    text = re.sub(replace_by_space_re," ",text) # replace symbols by space

    text = re.sub(bad_symbols_re, "",text) # remove bad symbols

    text = re.sub(links_re, "",text) # remove hyperlinks

    

    word_tokens = word_tokenize(text) # Creating word tokens out of the text

    

    filtered_tokens=[]

    for word in word_tokens:

        if word not in Stopwords:

            filtered_tokens.append(lemmatizer.lemmatize(word))

    

    text = " ".join(word for word in filtered_tokens)

    return text
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train["text"] = [text_prepare(x) for x in train["text"]]

test["text"] = [text_prepare(x) for x in test["text"]]
#train_input = bert_encode(train.text.values, tokenizer, max_len=160)

#test_input = bert_encode(test.text.values, tokenizer, max_len=160)



train_input = bert_encode(train["text"], tokenizer, max_len=160)

test_input = bert_encode(test["text"], tokenizer, max_len=160)



train_labels = train.target.values
train_input[2].shape
model = build_model(bert_layer, max_len=160)

model.summary()
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

submission.to_csv('submission_BERT.csv', index=False)