import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
df = pd.concat([train, test])
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

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

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)


df['text']=df['text'].apply(lambda x : remove_URL(x))
df['text']=df['text'].apply(lambda x : remove_html(x))
df['text']=df['text'].apply(lambda x: remove_emoji(x))
df['text']=df['text'].apply(lambda x : remove_punct(x))
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D, Input
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
# Create corpus that without stopwords
# Each input will be converted to a list of words
# Return corpus that contains all converted inputs

stop=set(stopwords.words('english'))

def create_corpus(df):
    corpus=[]
    for tweet in df['text']:
        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]
        corpus.append(words)
    return corpus

corpus=create_corpus(df)
# Import the pretrained GloVe, here is the 100-dimensional version
# Retrive the information in txt file
# Form a embedding dictionary, key is the word, and value is the corresponding embedding vector

embedding_dict={}
with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()
# The maximum length of each tweet is 50
MAX_LEN=50

# Tokenize the corpus, convert each input into tokens (each unique word will be represented by one token)
# The argument 'num_words' can be assigned to the 'Tokenizer' class
# It will keep the most common num_words-1 words based on word frequency
tokenizer=Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences=tokenizer.texts_to_sequences(corpus)

# If the original text was longer than 50, truncate at the end
# If the original text was shorter than 50, pad at the end
tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')

# word_index is a dictionary, the key is one unique word, the value is the corresponding token
word_index=tokenizer.word_index
num_words=len(word_index)+1
# Create embedding matrix
# Initialize the embedding matrix, here the first dimension is the number of unique words, second dimension is the dimension of the GloVe we chose
embedding_matrix=np.zeros((num_words,100))

for word,i in word_index.items():
    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec

# The embedding matrix represent each unique word in the corpus with a 1 by 100 vector.
# Model structure
# Layers: Embedding - Dropout - LSTM - Dense
# The output dimension of last layer is 1, since we are dealing binary classification here

model=Sequential()
embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False)

model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

optimzer=Adam(learning_rate=1e-5)
model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])

model.summary()
# Separate train and test data
train_X=tweet_pad[:train.shape[0]]
X_test=tweet_pad[train.shape[0]:]
train_y = train['target']

# Separate train and validation data
X_train,X_val,y_train,y_val=train_test_split(train_X,train_y,test_size=0.15)
print('Shape of train',X_train.shape)
print("Shape of Validation ",X_val.shape)


history = model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_val,y_val),verbose=2)
# Download the tokenizer from BERT
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
import tokenization
# Prepare the text that feeds to bert layer
# BERT needs some special format of the input, details here: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        input_sequence = ['[CLS]'] + text + ['[SEP]']
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0]*pad_len
        pad_masks = [1]*len(input_sequence) + [0]*pad_len
        segment_ids = [0]*max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

train_input = bert_encode(train.text.values, tokenizer, max_len=160)
test_input = bert_encode(test.text.values, tokenizer, max_len=160)
train_labels = train.target.values
# Model structure
# Layers: BERT - Dense

def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_word_ids')
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name='input_mask')
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    
    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
# Load BERT from Tfhub
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
model = build_model(bert_layer, max_len=160)
model.summary()
# Save the best model during training

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint],
    batch_size=16)
# Use the best model to predict

model.load_weights('model.h5')
test_pred = model.predict(test_input)
submission['target'] = test_pred.round().astype(int)
submission.to_csv('submission.csv', index=False)