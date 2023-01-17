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
# Official BERT tokenizer
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tensorflow_hub as hub
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer
from nltk.probability import FreqDist
import tokenization
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, LSTM, GRU, MaxPool1D, GlobalMaxPooling1D, Embedding, Dropout, SpatialDropout1D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import tensorflow as tf
import re
!pip install inflect
import inflect
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Append keyword to sentences
b_add_keyword = True

# Types of cleaning to apply - name helper functions accordingly
# cleaning = ['remove_url', 'remove_numbers', 'remove_unicode', 'remove_emoji', 'remove_stopwords']
# cleaning = ['remove_stopwords']
cleaning = []

max_len = 160  # 100
# Read data
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

print('n train/test: {}/{}'.format(train_df.shape[0], test_df.shape[0]))
print('n per class:\n{}'.format(train_df['target'].value_counts()))
train_df['location'].value_counts()
train_df['keyword'].isna().value_counts()
a = train_df['text']
train_df['text'].values
all_words = []
for t in train_df['text'].values:
    all_words.extend(wordpunct_tokenize(t))

counter = Counter(all_words)
counter.most_common(10)
def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_numbers(text):
    return re.sub(r'[^a-zA-Z\']', ' ', text)

def remove_unicode(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)
    
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

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    text = text.lower()
    
    text = " ".join([w for w in wordpunct_tokenize(text) if w not in stop_words])
    
    return text

# Convert number into words
p = inflect.engine()
def convert_number(text):
    # split string into list of words 
    temp_str = text.split() 
    # initialise empty list 
    new_string = [] 

    for word in temp_str: 
        # if word is a digit, convert the digit 
        # to numbers and append into the new_string list 
        if word.isdigit(): 
            temp = p.number_to_words(word) 
            new_string.append(temp) 

        # append the word as it is 
        else: 
            new_string.append(word) 

    # join the words of new_string to form a string 
    temp_str = ' '.join(new_string) 
    return temp_str

# longform_dict = {"don't": "do not", ""}
# def convert_to_longform(text):
    

# Word cloud
def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stop_words,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
# module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
# module_url = "https://tfhub.dev/google/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
# Define tokenizer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
def encode_text(texts, tokenizer, max_len):
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
    
train_text = train_df['text']
test_text = test_df['text']

# Add keyword
if b_add_keyword:
    for ii in range(train_text.shape[0]):
        if not pd.isna(train_df.loc[ii, 'keyword']):
            train_text[ii] = train_text[ii] + ' ' + train_df.loc[ii, 'keyword']
    
    for ii in range(test_text.shape[0]):
        if not pd.isna(test_df.loc[ii, 'keyword']):
            test_text[ii] = test_text[ii] + ' ' + test_df.loc[ii, 'keyword']

# Clean
for ii in cleaning:
    train_text = train_text.apply(eval(ii))
    test_text = test_text.apply(eval(ii))

all_words_train = []
for t in train_text.values:
    all_words_train.extend(wordpunct_tokenize(t))
all_words_test = []
for t in test_text.values:
    all_words_test.extend(wordpunct_tokenize(t))
counter = Counter(all_words_train)
print('train\n{}'.format(counter.most_common(10)))

counter = Counter(all_words_test)
print('test\n{}'.format(counter.most_common(10)))
wordpunct_tokenize("this is great don't"), word_tokenize("this is great don't"), tokenizer.tokenize("this is great don't")
print('train 0\n')
show_wordcloud(train_text.loc[train_df['target']==0])
print('train 1\n')
show_wordcloud(train_text.loc[train_df['target']==1])
train_text_enc = encode_text(train_text.values, tokenizer, max_len)
test_text_enc = encode_text(test_text.values, tokenizer, max_len)
# This model is the best so far without any cleaning!
def build_bert_model(bert_layer, max_len):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    
    # Using [CLS] token output from sequence output
    x = sequence_output[:, 0, :]  # use 0th output - belongs to CLS token - means classification
    
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    
    return model



"""
def build_bert_model(bert_layer, max_len):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    
    # Using [CLS] token output from sequence output
    x = sequence_output[:, 0, :]  # use 0th output - belongs to CLS token - means classification
#     x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    
    return model
"""

# Early stopping
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=3, verbose=1)

# Build model
m = build_bert_model(bert_layer, max_len)
m.summary()
train_perf = pd.DataFrame()
m.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])
fit_hist = m.fit(train_text_enc, train_df['target'].values, epochs=10, validation_split=0.2, batch_size=32)
train_perf = pd.concat((train_perf, pd.DataFrame(fit_hist.history)), ignore_index=True)
# Demo model to test out syntax
# in_l = Input(shape=(train_text_enc[0].shape[1], ))
# o = Dense(1, activation='sigmoid')(in_l)
# m = Model(inputs=in_l, outputs=o)

# # Early stopping
# es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1)

# m.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])
# fit_hist = m.fit(train_text_enc, train_df['target'].values, epochs=20, validation_split=0.2, batch_size=32, callbacks=[es])
# Plot training performance
plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.plot(train_perf['loss'])
plt.plot(train_perf['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.xlabel('epoch')
plt.title('loss')

plt.subplot(122)
plt.plot(train_perf['accuracy'])
plt.plot(train_perf['val_accuracy'])
plt.legend(['acc', 'val_acc'])
plt.xlabel('epoch')
plt.title('acc')
plt.show()
test_output = m.predict(test_text_enc)
test_pred = (test_output >= 0.5).astype(int)
submission = pd.DataFrame(data={'id': test_df['id'].values, 'target': test_pred.reshape(-1, )}) # {'id': test_df['id'].values.reshape(-1, 1), 'target': test_pred})
submission.head()

submission.to_csv(r'submission.csv', index=False)

model_score = pd.DataFrame(data={'id': test_df['id'].values, 'target': test_pred.reshape(-1, ), 'score': test_output.reshape(-1, )})
model_score.to_csv(r'model_score_bert.csv', index=False)

m.save('m_bert.h5')
