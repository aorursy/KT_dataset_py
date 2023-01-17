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
!pip install bert-for-tf2

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
# Loading pretrained bert layer
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)

# Loading tokenizer from the bert layer

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = BertTokenizer(vocab_file, do_lower_case)
# load the dataset

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
train_df = pd.read_csv("../input/nlp-getting-started/train.csv")
test_df = pd.read_csv("../input/nlp-getting-started/test.csv")

# text preprosessing

from nltk.stem import PorterStemmer #normalize word form
from nltk.probability import FreqDist #frequency word count
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords #stop words
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.probability import FreqDist 

def text_cleaning_hyperlink(text,rep):
    
    #remove hyper link
    return re.sub(r"http\S+","{}".format(rep),text) #remove hyperlink


def text_cleaning_punctuation(text):
    defined_punctuation = string.punctuation.replace('#','')  # specific innovation
    translator = str.maketrans( defined_punctuation, ' '*len( defined_punctuation)) #remove punctuation    
    return text.translate(translator)

def text_cleaning_stopwords(text):
    
    stop_words = set(stopwords.words('english'))
    word_token = word_tokenize(text)
    filtered_sentence = [w for w in word_token if not w in stop_words]
    return ' '.join(filtered_sentence) #return string of no stopwords


# convert all letters into lowercase ones
def text_cleaning_lowercase(text):
    return text.lower()

def text_extract(text_lst):
    
    txt = []
    for i,x in enumerate(text_lst):
        for j,p in enumerate(x):
            txt.append(p)
    return txt

# remove digits from the text

def remove_digits(txt):
    
    no_digits = ''.join(i for i in txt if not i.isdigit())
    return no_digits
from nltk import pos_tag, word_tokenize
ps = PorterStemmer()
wnl = WordNetLemmatizer()


def word_lemmatizer_1(sentence):
    
    n_sentence = []
    for word, tag in pos_tag(word_tokenize(sentence)):
        wntag = tag[0].lower()
        if wntag in ['a', 'r', 'n', 'v']:
            wntag = wntag
        else:
            wntag = None
        if not wntag:
            lemma = wnl.lemmatize(word)
        else:
            lemma = wnl.lemmatize(word,wntag)
        n_sentence.append(lemma)
    n_s = ' '.join(n_sentence)
    return n_s
from nltk import pos_tag, word_tokenize
ps = PorterStemmer()
wnl = WordNetLemmatizer()


def word_lemmatizer_2(sentence):
    
    n_sentence = []
    for word, tag in pos_tag(word_tokenize(sentence)):
        wntag = tag[0].lower()
        if wntag in ['a', 'r', 'n']:
            wntag = wntag
        else:
            wntag = None
        if not wntag:
            lemma = wnl.lemmatize(word)
        else:
            lemma = wnl.lemmatize(word,wntag)
        n_sentence.append(lemma)
    n_s = ' '.join(n_sentence)
    return n_s
import re
import string

#print(string.punctuation.replace('#',''))

#Clean the text
train_df['text_clean'] = train_df.text.apply(lambda x: text_cleaning_hyperlink(remove_digits(x),'http'))
train_df['text_clean'] = train_df['text_clean'].apply(lambda x:  text_cleaning_punctuation(x) )
train_df['text_clean'] = train_df['text_clean'].apply(lambda x:  text_cleaning_stopwords(x) )

#train_df['text_clean'] = train_df.text.apply(lambda x: text_cleaning_stopwords(text_cleaning_punctuation(text_cleaning_hyperlink(remove_digits(x),'http'))))
# train_df['text_clean'] = train_df['text_clean'].apply(lambda x: list(set(x.split(' '))))
# train_df['text_clean'] = train_df['text_clean'].apply(lambda x: ' '.join(x))

test_df['text_clean'] = test_df.text.apply(lambda x: text_cleaning_hyperlink(remove_digits(x),'http'))
test_df['text_clean'] = test_df['text_clean'].apply(lambda x:  text_cleaning_punctuation(x) )
test_df['text_clean'] = test_df['text_clean'].apply(lambda x:  text_cleaning_stopwords(x) )



# see a sample of encoding
text = train_df.text[0]

#tokenize 
tokens_list = tokenizer.tokenize(text)
print("Text after tokenization like this:", tokens_list)

#initialize dimension
max_len = 25
text = tokens_list[:max_len-2]
input_sequence = ["[CLS]"] + text +["[SEP]"]
print("Text after adding flag -[ClS] and [SEP]: ", input_sequence )


tokens = tokenizer.convert_tokens_to_ids(input_sequence)
print("tokens to their id: ", tokens)

pad_len = max_len  - len(input_sequence)
tokens += [0] * pad_len
print("tokens: ", tokens)

print(pad_len)
pad_masks = [1] * len(input_sequence) + [0] * pad_len
print("Pad Masking: ", pad_masks)

segment_ids = [0] * max_len
print("Segment Ids: ",segment_ids)
# Function to encoe the text into tokens, mask, and segment flags

import numpy as np


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

MAX_LEN = 160

# encode train set 
# train_input = bert_encode(train_df['text_clean'], tokenizer, max_len=MAX_LEN)

# encode test set 
# test_input = bert_encode(test_df['text_clean'], tokenizer, max_len= MAX_LEN )


from tensorflow.keras.layers import Input



input_word_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")
input_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_mask")
segment_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="segment_ids")

    
#output

from tensorflow.keras.layers import Dense

_, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
clf_output = sequence_output[:, 0, :]
out = Dense(1, activation='sigmoid')(clf_output)

#model initialization 

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


from sklearn.model_selection import train_test_split


# personal test
# X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(train_df['text_clean'], train_df.target, test_size=0.3)
# train_input = bert_encode(X_train_1, tokenizer, max_len=MAX_LEN)
# test_input = bert_encode(X_test_1, tokenizer, max_len=MAX_LEN)
# train_labels = y_train_1
# test_labels = y_test_1

# submission test

# encode train set 
train_input = bert_encode(train_df.text.values, tokenizer, max_len=MAX_LEN)
# encode  test set 
test_input = bert_encode(test_df.text.values, tokenizer, max_len= MAX_LEN )

train_labels = train_df.target.values

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    batch_size=32
)

model.save('model.h5')


# train
# train_history = model.fit(
#     train_input, train_labels,
#     validation_split=0.2,
#     epochs=20,
#     batch_size=32
# )

# model.save('model.h5')
test_pred = model.predict(test_input)
preds = test_pred.round().astype(int)
preds

sample_submission["target"] = preds
sample_submission.target.value_counts()
sample_submission.to_csv("submission.csv", index = False)

# scores = model.evaluate(test_input, test_labels, verbose=1)
# print("Accuracy:", scores[1])
