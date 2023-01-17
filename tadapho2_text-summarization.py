import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path = "/kaggle/input/amazon-fine-food-reviews/Reviews.csv"
df = pd.DataFrame()
df = pd.read_csv(path)
df.shape
df = df.sample(400000)
df.shape
df = df[['Text', 'Summary']]
df.dropna(axis=0, inplace=True)                    
df.drop_duplicates(subset=['Summary'], inplace=True)  
df.reset_index(drop=1, inplace=True)
df.head(5)
df.shape
print(df['Text'][0])
print(df['Summary'][0])
import spacy
import nltk
import re
# nlp = spacy.load('en_core_web_sm')
# nlp = spacy.load('en_core_web_lg')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = nltk.corpus.stopwords.words('english')
# nlp = spacy.load('en', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "can not", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
print(stop_words)
print(contraction_mapping)
print(punctuations)
def cleansing_text(text):
    text = text.lower() # Convert to lowercase
    text = re.sub('<pre>.*?</pre>', '', text, flags = re.DOTALL)  # Remove HTML tags
    text = re.sub('<code>.*?</code>', '', text, flags = re.DOTALL)
    text = re.sub('<[^>]+>', '',text ,flags = re.DOTALL)
    text = ' '.join([contraction_mapping[i] if i in contraction_mapping else i for i in text.split(" ")]) # Contraction mapping 
    text = re.sub(r"'s\b", "", text)  # Remove (‘s) 
    text = re.sub("[^a-zA-Z]" ," ", text) # Remove punctuations and special characters
    text = ' '.join([i for i in text.split() if i not in punctuations]) # Remove punctuations
    text = ' '.join([i for i in text.split() if i not in stop_words]) # Remove stop_words
#     text = ''.join([str(doc) for doc in nlp.pipe(text, batch_size = 5000, n_threads=-1)])
    return text

def cleansing_summary(summary):
    summary = summary.lower() # Convert to lowercase
    summary = re.sub('<pre>.*?</pre>', '', summary, flags = re.DOTALL)  # Remove HTML tags
    summary = re.sub('<code>.*?</code>', '', summary, flags = re.DOTALL)
    summary = re.sub('<[^>]+>', '',summary ,flags = re.DOTALL)
    summary = ' '.join([contraction_mapping[i] if i in contraction_mapping else i for i in summary.split(" ")]) # Contraction mapping 
    summary = re.sub(r"'s\b", "", summary)  # Remove (‘s) 
    summary = re.sub("[^a-zA-Z]" ," ", summary) # Remove punctuations and special characters
    summary = ' '.join([i for i in summary.split() if i not in punctuations]) # Remove personal punctuations
    summary = ' '.join([i for i in summary.split() if i not in stop_words]) # Remove stop_words
#     summary = ''.join([str(doc) for doc in nlp.pipe(summary, batch_size = 5000, n_threads=-1)])
#     summary = 'START_ ' + str(summary) + ' END_'
    return summary
from tqdm.notebook import tqdm

texts = []
for text in tqdm(df['Text']):
    texts.append(cleansing_text(text))
df['Text_Cleaned'] = texts  
print("::::: Text_Cleaned :::::")
print(df['Text_Cleaned'][0:5], "\n")

summaries = []
for text in tqdm(df['Summary']):
    summaries.append(cleansing_summary(text))
df['Summary_Cleaned'] =  summaries 
print("::::: Summary :::::")
print(df['Summary_Cleaned'][0:5], "\n")

corpus = list(df['Text_Cleaned'])
print(df['Text_Cleaned'][0])
print(df['Summary_Cleaned'][0])
text_count = []
summary_count = []

for sent in df['Text_Cleaned']:
    text_count.append(len(sent.split()))
for sent in df['Summary_Cleaned']:
    summary_count.append(len(sent.split()))

graph_df = pd.DataFrame()
graph_df['text'] = text_count
graph_df['summary'] = summary_count
graph_df['text'].describe()
graph_df['summary'].describe()
graph_df['text'].hist(bins = 25, range=(0, 200))
plt.show()
graph_df['summary'].hist(bins = 15, range=(0, 15))
plt.show()
# Check how much % of text have 10-100 words
count = 0
for i in graph_df['text']:
    if i > 10 and i <= 100:
        count = count + 1
print(count / len(graph_df['text']))
# Check how much % of summary have 2-10 words
count = 0
for i in graph_df['summary']:
    if i > 1 and i <= 10:
        count = count + 1
print(count / len(graph_df['summary']))

# Model to summarize  
# 11 - 100 words for Text
# 2 - 10 words for Summary 

max_text_len = 100
max_summary_len = 10

cleaned_text = np.array(df['Text_Cleaned'])
cleaned_summary = np.array(df['Summary_Cleaned'])

short_text = []
short_summary = []

for i in range(len(cleaned_text)):
    if(len(cleaned_summary[i].split()) <= max_summary_len 
       and len(cleaned_summary[i].split()) > 1 
       and len(cleaned_text[i].split()) <= max_text_len 
       and len(cleaned_text[i].split()) > 10):
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])
        
post_pre = pd.DataFrame({'text':short_text,'summary':short_summary})
# Add sostok and eostok
post_pre['summary'] = post_pre['summary'].apply(lambda x : 'sostok '+ x + ' eostok')
post_pre.shape
post_pre
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

# train test split
x_tr,x_test,y_tr,y_test = train_test_split(np.array(post_pre['text']),
                                         np.array(post_pre['summary']),
                                         test_size = 0.2,
                                         random_state = 0,
                                         shuffle = True)
# train validation split
x_tr,x_val,y_tr,y_val = train_test_split(x_tr,
                                         y_tr,
                                         test_size = 0.2,
                                         random_state = 0,
                                         shuffle = True)
x_tr.shape
x_test.shape
x_val.shape
# Tokenize text to get the vocab count
#prepare a tokenizer for training data
x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(x_tr))

#prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer()   
y_tokenizer.fit_on_texts(list(y_tr))
thresh = 6
cnt = 0
# tot_cnt = 0
tot_cnt = len(x_tokenizer.word_counts)
freq = 0
tot_freq = 0

keys = []
values = []

for key,value in x_tokenizer.word_counts.items():
    keys.append(key)
    values.append(value)
    if(value < thresh):
        cnt = cnt + 1

df_frequency = pd.DataFrame({'word':keys,'frequency':values})
df_frequency.sort_values(by='frequency', ascending=False, inplace=True)
df_frequency.reset_index(inplace=True, drop=0)
df_frequency
print("% Rare words in vocabulary:",(cnt / tot_cnt) * 100)
tot_cnt, cnt
fig, ax = plt.subplots(figsize=(6,10), ncols=1, nrows=1)
sns.barplot(x='frequency',y='word',data=df_frequency[:20], palette='Reds_r', ax=ax);
#prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer(num_words = tot_cnt - cnt) 
x_tokenizer.fit_on_texts(list(x_tr))

#convert text sequences into integer sequences (i.e one-hot encodeing all the words)
x_tr_seq    =   x_tokenizer.texts_to_sequences(x_tr) 
x_val_seq   =   x_tokenizer.texts_to_sequences(x_val)
x_test_seq = x_tokenizer.texts_to_sequences(x_test)

#padding zero upto maximum length
x_tr    =   pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')
x_val   =   pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')
x_test = pad_sequences(x_test_seq, maxlen=max_text_len, padding='post')

#size of vocabulary ( +1 for padding token)
x_voc   =  x_tokenizer.num_words + 1

print("Size of vocabulary in X = {}".format(x_voc))
thresh = 3
cnt = 0
tot_cnt = len(y_tokenizer.word_counts)
freq = 0
tot_freq = 0

keys = []
values = []

for key,value in y_tokenizer.word_counts.items():
    keys.append(key)
    values.append(value)
    if(value < thresh):
        cnt = cnt + 1

df_frequency = pd.DataFrame({'word':keys,'frequency':values})
df_frequency.sort_values(by='frequency', ascending=False, inplace=True)
df_frequency.reset_index(inplace=True, drop=0)
df_frequency
print("% Rare words in vocabulary:",(cnt / tot_cnt) * 100)
tot_cnt, cnt
print("% Rare words in vocabulary:",(cnt / tot_cnt) * 100)
tot_cnt, cnt
fig, ax = plt.subplots(figsize=(6,10), ncols=1, nrows=1)
sns.barplot(x='frequency',y='word',data=df_frequency[3:20], palette='Reds_r', ax=ax);
#prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer(num_words = tot_cnt-cnt) 
y_tokenizer.fit_on_texts(list(y_tr))

#convert text sequences into integer sequences (i.e one hot encode the text in Y)
y_tr_seq    =   y_tokenizer.texts_to_sequences(y_tr) 
y_val_seq   =   y_tokenizer.texts_to_sequences(y_val) 
y_test_seq = y_tokenizer.texts_to_sequences(y_test) 

#padding zero upto maximum length
y_tr    =   pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
y_val   =   pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')
y_test = pad_sequences(y_test_seq, maxlen=max_summary_len, padding='post')

#size of vocabulary
y_voc  =   y_tokenizer.num_words +1
print("Size of vocabulary in Y = {}".format(y_voc))
from tensorflow.keras.backend import clear_session
import gensim
from numpy import *
import numpy as np
import pandas as pd 
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
print(f"Size of vocabulary from the w2v model = {x_voc}")

clear_session()

latent_dim = 256
embedding_dim = 128

# Encoder
encoder_inputs = Input(shape=(max_text_len,))

#embedding layer
enc_emb =  Embedding(x_voc, embedding_dim,trainable=True)(encoder_inputs)

#encoder lstm 1
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

#encoder lstm 2
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

#encoder lstm 3
encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

#embedding layer
dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

#dense layer
decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)

history=model.fit([x_tr,y_tr[:,:-1]], 
                  y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:],
                  epochs=10,
                  callbacks=[es],
                  batch_size=128, 
                  validation_data=([x_val,y_val[:,:-1]], 
                                   y_val.reshape(y_val.shape[0],
                                                 y_val.shape[1], 
                                                 1)[:,1:]))
from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index

# Encode the input sequence to get the feature vector
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs) 
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs2) 

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence
def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString
totle = []
totle_predicted = []
accuracy = []

# sample 5000 test
for i in tqdm(range(0, 5000)):
    review = seq2text(x_test[i])
    original_summary = seq2summary(y_test[i])
    predicted_summary = decode_sequence(x_test[i].reshape(1, max_text_len))
    print("Review:", review)
    print("Original summary:", original_summary)
    print("Predicted summary:", predicted_summary)
    
#     if len(original_summary.split()) != 0:
    count = 0
    for j in predicted_summary.split():
        if j in review:
            count += 1
#     count = 0
#     for k in decode_sequence(x_tr[i].reshape(1, max_text_len)).split():
#         if k in original_summary:
#             count += 1
    totle.append(len(predicted_summary.split()))
    accuracy.append(count/len(predicted_summary.split()))
    print(f"{count} / {len(predicted_summary.split())}")
    print("\n")
sum(accuracy)/len(accuracy)