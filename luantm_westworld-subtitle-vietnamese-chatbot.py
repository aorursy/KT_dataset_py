!pip install srt
import re
import numpy as np # linear algebra
import pandas as pd 
import os
print(os.listdir("../input"))
import tensorflow as tf
import gensim
from nltk import ngrams
import srt 
def get_text(item):
    #remove special charaters
    item = re.sub('\-|\.|\,|\:|\#|\[|\]|<[^>]+>', ' ', item)
    item = item.replace('?', ' ?').strip()
    return item
#Create data 
def create_data():
    x = []
    y = []
    for file in os.listdir('../input/westworld-subtitles'):
        f = open(os.path.join('../input/westworld-subtitles', file),encoding='utf-16').read()
        subs = list(srt.parse(f))
        for idx, sub in enumerate(subs):
            if idx>0 and (subs[idx].start - subs[idx - 1].end).seconds < 15: # >15s : other conversation
                x.append(get_text(subs[idx-1].content))
                y.append(get_text(subs[idx].content))
                
            
    return x, y
    
x, y = create_data()
print(x[0:10])
print(y[0:10])
from gensim.models import KeyedVectors
model_path = '../input/w2vmodelvn-word2vec-vn-gensim/W2VModelVN.bin'
w2v = KeyedVectors.load_word2vec_format(model_path, fvocab=None, binary=True, encoding='utf8')
print(w2v.similar_by_word('anh'))

keys = list(w2v.vocab.keys())
print(keys[0:400])
vectors = w2v.vectors
print(vectors[0:10])
# print(w2v.vocab.keys())
#TODO Tokenize with keys
def tokenize(s):
    result = []
    arr = s.split()
    for item in arr:
        if len(result) > 0:
            tmp = result[-1] + '_' + item
            if tmp in keys:
                result[-1] = tmp
            else:
                result.append(item)
        else:
            result.append(item)
            
    return result
print(tokenize('Nhân viên trong không gian thật thân thiện'))
#Create models
x_tok = [tokenize(item) for item in x]
y_tok = [tokenize(item) for item in y]
print(x_tok[0:10])
max_len = 0
for item in y_tok:
    if len(item) > max_len:
        max_len = len(item)
print(max_len)
vectors = w2v.vectors
print(vectors.shape)
keys = list(w2v.vocab.keys())
vectors = w2v.vectors
PAD = np.ones((1,300))
UNK = np.zeros((1,300))
# print(PAD)
keys.append('<PAD>')
vectors = np.append(vectors, PAD, axis=0)
keys.append('<UNK>')
vectors = np.append(vectors, UNK, axis=0)
def get_vector_by_key(key):
    if key not in keys:
        key = '<UNK>'
    return vectors[keys.index(key)]
get_vector_by_key('Thân_thiện')
x_train = []
y_train = []
SENTENCE_LEN = 25
for toks in x_tok:
    tmp = [get_vector_by_key(words) for words in toks]
    tmp += [get_vector_by_key('<PAD>')] * (25 - len(tmp))
    if len(tmp) == 25:
        x_train.append(tmp)
for toks in y_tok:
    toks = ['<PAD>'] + toks
    tmp = [get_vector_by_key(words) for words in toks]
    tmp += [get_vector_by_key('<PAD>')] * (25 - len(tmp))
    if len(tmp) == 25:
        y_train.append(tmp)
    
x_train = np.array(x_train).reshape(-1, 25, 300)
y_train = np.array(y_train).reshape(-1, 25, 300)
print(x_train.shape)
print(y_train.shape)
lstm = tf.nn.rnn_cell.LSTMCell(128)
dense = tf.layers.Dense(300)
init_state = state = lstm.zero_state(x_train.shape[0], dtype=tf.float32)
x = x_train
y = y_train
state = init_state
outputs = []
# x_arr = 
for i in range(x.shape[1]):
    xi = x[:, i, :]
    xi = tf.convert_to_tensor(xi, dtype=tf.float32)
    output, state = lstm(xi, state)
    logits = dense(output)
    outputs.append(logits)
outputs = tf.transpose(tf.convert_to_tensor(outputs), perm=[1, 0, 2])
loss_op = tf.losses.mean_squared_error(
    predictions=outputs, labels=y)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(10000):
    sess.run(train_op)
    loss = sess.run(loss_op)
    print(f'epoch: {epoch}      loss: {loss}')

print(x_tok[0])
print(x_tok[1])
print(x_tok[0][0])
# print(x_tok[0][0])
test = get_vector_by_key(x_tok[0][0])
# print(keys.index(x_tok[0][0]))
# print(test)
# print(vectors[7258])
# print((test - x[0][0]) ** 2)
su = np.sum((vectors - x[0][0]) ** 2, axis=1)
# print(np.argmin(su))
# print()
# print(su[np.argmin(su)])
output = sess.run(outputs)
print(output.shape)

for j in range(10):
    print(y_tok[j])
    for i in range(25):
        t = np.argmin(np.sum((vectors - output[j][i]) ** 2, axis=1))
        if keys[t] != '<PAD>':
            print(keys[t], end=' ')
    print()

