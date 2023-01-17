import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, GRU, Input, Flatten
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from gensim.models import Word2Vec, KeyedVectors, word2vec
import gensim
import torch
import itertools
import os
import string
from nltk.tokenize import word_tokenize
from keras.utils import to_categorical
import tqdm

EMBD_DIM =100
path = '/kaggle/input/isods2020/'
os.listdir(path)
train_data = pd.read_csv(path + 'training_data_sentiment.csv')
test_data = pd.read_csv(path + 'testing_data_sentiment.csv')
print('Size of train_data: ',train_data.shape)
print('Size of test_data: ',test_data.shape)
train_data.head()
test_data.head()
train_data['is_unsatisfied'].value_counts()
train_data['is_unsatisfied'].hist()
def Q_and_A(idx, data):
    print(train_data.question[idx])
    print(train_data.answer[idx])
    q = train_data.question[idx].split('|||')
    a = train_data.answer[idx].split('|||')
    if q[0] == '': q = q[1:]
    print(len(q) , '-' , len(a))
    for i in range(min(len(q),len(a))):
        print(q[i])
        print(a[i])
        print('\n')
Q_and_A(1, train_data)
Q_and_A(2,train_data)
def clean_text(text):
    
    table = str.maketrans('','',string.punctuation)
    text = text.lower().split()
    text = [word.translate(table) for word in text]
    text = ' '.join(text)
    
    return text

def clean_data(messages):
    
    messages = messages.split('|||')
    messages = map(clean_text,messages)
    messages = '|||'.join(messages)
    
    return messages
train_data['Where'] = 'train'
test_data['Where'] ='test'
data = train_data.append(test_data)
data
data['clear_ques'] = [clean_data(x) for x in data.question]
data['join_ques'] = [x.replace('|||',' ') for x in data.clear_ques]
data['len_ques'] = [len(x.split('|||'))/10 for x in data.clear_ques]
data['max_word_ques'] = [max([len(words.split()) for words in x.split('|||')])/100 for x in data.clear_ques ]
data['min_word_ques'] = [min([len(words.split()) for words in x.split('|||')])/100 for x in data.clear_ques ]
data['mean_word_ques'] = [sum([len(words.split()) for words in x.split('|||')])/len([words for words in x.split('|||')])/100 for x in data.clear_ques]
                         
data['clear_ans'] = [clean_data(x) for x in data.answer]
data['join_ans'] = [x.replace('|||',' ') for x in data.clear_ans]
data['len_ans'] = [len(x.split('|||'))/10 for x in data.clear_ans]
data['max_word_ans'] = [max([len(words.split()) for words in x.split('|||')])/100 for x in data.clear_ans ]
data['min_word_ans'] = [min([len(words.split()) for words in x.split('|||')])/100 for x in data.clear_ans ]
data['mean_word_ans'] = [sum([len(words.split()) for words in x.split('|||')])/len([words for words in x.split('|||')])/100 for x in data.clear_ans]
data['Q_A_len_diff'] = list((np.array(data['len_ques'])- np.array(data['len_ans']))/(np.array(data['len_ques'])+np.array(data['len_ans'])/2))                        
data['label'] = data['is_unsatisfied'].apply(lambda x: 1 if x == 'Y' else 0)
# Get max length sentences
max_len_q = max((data['max_word_ques']*100).values.tolist())
max_len_a = max((data['max_word_ans']*100).values.tolist())
max_len = max(max_len_q,max_len_a)
train_data = data[data.Where == 'train']
test_data = data[data.Where == 'test']
del data
print('Size of train_data: ',train_data.shape)
print('Size of test_data: ',test_data.shape)
train_data[['clear_ques','join_ques']].tail(10)
X_ques = train_data['join_ques'].values
X_ans = train_data['join_ans'].values
y_train = train_data['label'].values

X_ques_test = test_data['join_ques'].values
X_ans_test = test_data['join_ans'].values
y_test = test_data['label'].values
# X = train_data[['join_ques','join_ans','len_ques','max_word_ques','min_word_ques', 'mean_word_ques',
#              'len_ans', 'max_word_ans', 'min_word_ans', 'mean_word_ans','Q_A_len_diff']]
# X = train_data[['join_ques','join_ans']]
# X = train_data['ques_ans']
# y = train_data['label']
# X.shape, y.shape
#word2vec model
model_ques = gensim.models.Word2Vec(sentences = X_ques.tolist(),size=EMBD_DIM,workers=2,min_count=1)
model_ans = gensim.models.Word2Vec(sentences = X_ans.tolist(),size=EMBD_DIM,workers=2,min_count=1)
model_ques_test = gensim.models.Word2Vec(sentences = X_ques_test.tolist(),size=EMBD_DIM,workers=2,min_count=1)
model_ans_test = gensim.models.Word2Vec(sentences = X_ans_test.tolist(),size=EMBD_DIM,workers=2,min_count=1)
#vocab size each of model
words_ques =list(model_ques.wv.vocab)
words_ans =list(model_ans.wv.vocab)
words_ques_test =list(model_ques_test.wv.vocab)
words_ans_test =list(model_ans_test.wv.vocab)

print('Vocabulary question size %d' % len(words_ques))
print('Vocabulary answer size %d' % len(words_ans))
print('Vocabulary question size %d' % len(words_ques_test))
print('Vocabulary answer size %d' % len(words_ans_test))
# Vocabulary of train dataset
vocab_train = list(set(words_ques).union(set(words_ans)))
vocab_test =  list(set(words_ques_test).union(set(words_ans_test)))
print('Vocabulary train size %d' % len(vocab_train))
print('Vocabulary test size %d' % len(vocab_test))
vocab_all =  list(set(vocab_train).union(set(vocab_test)))
print('All vocabulary size %d' % len(vocab_all))
#save model
filename_q = 'q_emb_word2vec.txt'
filename_a = 'a_emb_word2vec.txt'
filename_q_test = 'q_test_emb_word2vec.txt'
filename_a_test = 'a_test_emb_word2vec.txt'
model_ques.wv.save_word2vec_format(filename_q,binary=False)
model_ans.wv.save_word2vec_format(filename_a,binary=False)
model_ques_test.wv.save_word2vec_format(filename_q_test,binary=False)
model_ans_test.wv.save_word2vec_format(filename_a_test,binary=False)
def load_embedding(file_name):
    embeddings_index = {}
    f = open(os.path.join('',file_name),encoding ='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        if  word == ' ':
            continue  
        coefs = np.asarray(values[1:])
        embeddings_index[word] =coefs
    f.close
    return embeddings_index
    
    

train_q_emb = load_embedding('q_emb_word2vec.txt')
train_a_emb = load_embedding('a_emb_word2vec.txt')
test_q_emb = load_embedding('q_test_emb_word2vec.txt')
test_a_emb = load_embedding('a_test_emb_word2vec.txt')

# Join embedding dicts into unique embedding dict
list_embs = [train_q_emb,train_a_emb,test_q_emb,test_a_emb]
embeddings_index = list_embs[0]
for i in range(1,len(list_embs)):
    embeddings_index = dict(list_embs[i],**embeddings_index)

# vectorize the text samples into a 2D integer tensor
nb_classes = 2

tokenizer = Tokenizer(num_words=len(embeddings_index), split=' ')
tokenizer.fit_on_texts(X_ques.tolist()+X_ans.tolist()+X_ques_test.tolist()+X_ans_test.tolist())

sequences_train_q = tokenizer.texts_to_sequences(X_ques.tolist())
sequences_train_a = tokenizer.texts_to_sequences(X_ans.tolist())
sequences_test_q = tokenizer.texts_to_sequences(X_ques_test.tolist())
sequences_test_a = tokenizer.texts_to_sequences(X_ans_test.tolist())
#pad sequences

word_index = tokenizer.word_index
print('Found %s unique tokens.'%len(word_index))
train_q_pad = pad_sequences(sequences_train_q, maxlen =int(max_len))
train_a_pad = pad_sequences(sequences_train_a, maxlen =int(max_len))
test_q_pad = pad_sequences(sequences_test_q, maxlen =int(max_len))
test_a_pad = pad_sequences(sequences_test_a, maxlen =int(max_len))

train_pad = np.concatenate((train_q_pad,train_a_pad),axis = 1)
test_pad = np.concatenate((test_q_pad,test_a_pad),axis = 1)

y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)
print('Shape of train tensor',train_pad.shape)
print('Shape of train sentiment tensor:',y_train.shape)
print('Shape of test tensor',test_pad.shape)
print('Shape of test sentiment tensor:',y_test.shape)
# Convert into embedding matrix
num_words = len(word_index) +1
embedding_matrix = np.zeros((num_words,EMBD_DIM))
for word, i in word_index.items():
    if i > num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
    #Words not found in embedding index wil be all-zeros
        embedding_matrix[i] = embedding_vector
print(num_words)
#embeddings_initializer = Constant(embedding_matrix),input_length=int(max_len),
#n_timesteps, n_features= train_pad.shape[0], train_pad.shape[1]
input_layer = Input(shape=(int(train_pad.shape[1])))
embedding_layer = Embedding(num_words,
                           EMBD_DIM,
                            embeddings_initializer = Constant(embedding_matrix),
                           trainable=False,)
embedding_sequences = embedding_layer(input_layer)
LSTM_layer = LSTM(units=32,dropout=0.2, recurrent_dropout=0.2,return_sequences=True)(embedding_sequences)
flatten_layer = Flatten()(LSTM_layer)
output_layer = Dense(nb_classes,activation='softmax')(flatten_layer)

model = Model(input_layer,output_layer)
model.summary()

opt = Adam(learning_rate=1e-3)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics = [AUC()])
#es = EarlyStopping(verbose=1,patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
print(y_train.shape)
print(train_pad.shape)
model.fit(train_pad,y_train,validation_split=0.2,batch_size = 32,epochs=5,callbacks=[reduce_lr])
def find_best_fixed_threshold(preds, targs, do_plot=False):
    score = []
    thrs = np.arange(0, 0.95, 0.01)
    for thr in tqdm(thrs):
        score.append(f1_score(targs, (preds >= thr).astype(int), average='macro' ))
    score = np.array(score)
    pm = score.argmax()
    best_thr, best_score = thrs[pm], score[pm].item()
    print('thr= ', best_thr, ' F1= ', best_score)
    return best_thr, best_score
test_pred = model.predict(test_pad)
best_thr, best_score = find_best_fixed_threshold(test_pred,y_test,do_plot=False)
# test_pred = model.predict(test_pad)
# submission.to_csv("submission.csv", index = False)
# submission.head()