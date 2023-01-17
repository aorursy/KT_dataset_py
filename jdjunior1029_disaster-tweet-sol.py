# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# !pip install nltk --upgrade

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
import nltk
from nltk.tokenize import word_tokenize
from nltk import RegexpTokenizer
import csv
from nltk.corpus import stopwords

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
        
retokenize = RegexpTokenizer("[\w]+")

en_stopws = stopwords.words('english')  # this loads the default stopwords list for English
# en_stopws.append('spam')              # add any words you don't like to the list

train_file_path = '/kaggle/input/nlp-getting-started/train.csv'
train_file = open(train_file_path, 'r', encoding='utf-8')
train_file_csv = csv.reader(train_file)

val_file_path = '/kaggle/input/nlp-getting-started/test.csv'
val_file = open(val_file_path, 'r', encoding='utf-8')
val_file_csv = csv.reader(val_file)

train = []
test = []
val = []
val_id = []
vocab = {}
max_word_num_in_sent = 0
len_data = 7613
train_len = int(7613 * 0.8)
print(len_data, train_len, len_data - train_len)

for idx, data in enumerate(train_file_csv):
    if idx == 0:
        continue
    # id, keyword, location, text, target
    sentence = []
    for i in range(1, 4):
        if data[i]:
            sentence.extend(retokenize.tokenize(data[i]))
    result = [w.lower() for w in sentence if w.lower() not in en_stopws and len(w) > 2]
    for word in result:
        if word not in vocab:
            vocab[word] = 0
        vocab[word] += 1
    y_gt = data[4]
    # print(tokens, y_gt)
    if idx <= train_len:
        train.append((result, y_gt))
    else:
        test.append((result, y_gt))
    # print(result)
    
    if max_word_num_in_sent < len(result):
        max_word_num_in_sent = len(result)
        
for idx, data in enumerate(val_file_csv):
    if idx == 0:
        continue
    sentence = []
    val_id.append(int(data[0]))
    for i in range(1, 4):
        if data[i]:
            sentence.extend(retokenize.tokenize(data[i]))
    result = [w.lower() for w in sentence if w.lower() not in en_stopws and len(w) > 2]
    for word in result:
        if word not in vocab:
            vocab[word] = 0
        vocab[word] += 1
    val.append(result)
    # print(result)
    
    if max_word_num_in_sent < len(result):
        max_word_num_in_sent = len(result)
    
train_file.close()
val_file.close()
    
np.save('train_data', train)
np.save('test_data', test)
np.save('val_data', val)
np.save('val_id', val_id)

print(f'max_word_num_in_sent: {max_word_num_in_sent}, len(vocab): {len(vocab)}')  # 26

# Any results you write to the current directory are saved as output.
from tensorflow.keras.preprocessing import sequence


vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)
word_to_index = {}
i=0
for (word, frequency) in vocab_sorted :
    if frequency > 1 : # 정제(Cleaning) 챕터에서 언급했듯이 빈도수가 적은 단어는 제외한다.
        i=i+1
        word_to_index[word] = i
# print(word_to_index)
print(f'max_features(len(word_to_index)): {len(word_to_index)}')

train_data = np.load('train_data.npy', allow_pickle=True)
test_data = np.load('test_data.npy', allow_pickle=True)
val_data = np.load('val_data.npy', allow_pickle=True)

print(f'Integer Encoding...')
x_train = []
x_test = []
x_val = []
y_train = []
y_test = []
for sentence, label in train_data:
    encoded_sentence = []
    for word in sentence:
        if word in word_to_index:
            encoded_sentence.extend([word_to_index[word]])
    x_train.append(encoded_sentence)
    if int(label) == 1:
        y_train.append([0, 1])
    else:
        y_train.append([1, 0])
for sentence, label in test_data:
    encoded_sentence = []
    for word in sentence:
        if word in word_to_index:
            encoded_sentence.extend([word_to_index[word]])
    x_test.append(encoded_sentence)
    if int(label) == 1:
        y_test.append([0, 1])
    else:
        y_test.append([1, 0])
for sentence in val_data:
    encoded_sentence = []
    for word in sentence:
        if word in word_to_index:
            encoded_sentence.extend([word_to_index[word]])
    x_val.append(encoded_sentence)
        
print(len(x_train), len(y_train), len(x_test), len(y_test), len(x_val))

print(f'x_train shape: {np.shape(x_train)}, x_train[0] shape: {np.shape(x_train[0])}, type: {type(x_train)}')
print('y_train shape:', np.shape(y_train))
print('x_test shape:', np.shape(x_test))
print('y_test shape:', np.shape(y_test))
print('x_val shape:', np.shape(x_val))

print(f'Pad sequences (samples x time)... by maxlen(max_word_num_in_sent)={max_word_num_in_sent}')
x_train_pad = sequence.pad_sequences(x_train, maxlen=max_word_num_in_sent)
x_test_pad = sequence.pad_sequences(x_test, maxlen=max_word_num_in_sent)
x_val_pad = sequence.pad_sequences(x_val, maxlen=max_word_num_in_sent)
    
x_train_pad = np.asarray(x_train_pad)
y_train = np.asarray(y_train)
x_test_pad = np.asarray(x_test_pad)
y_test = np.asarray(y_test)
x_val_pad = np.asarray(x_val_pad)

print(f'x_train shape: {x_train_pad.shape}, x_train[0] shape: {np.shape(x_train_pad[0])}, type: {type(x_train_pad)}')
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test_pad.shape)
print('y_test shape:', y_test.shape)
print('x_val shape:', x_val_pad.shape)

np.save('x_train', x_train_pad)
np.save('y_train', y_train)
np.save('x_test', x_test_pad)
np.save('y_test', y_test)
np.save('x_val', x_val_pad)

print('encoded data saved')
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate, Dropout, Input, Concatenate, Conv1D, Bidirectional, LSTM


class FastText(Model):

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num=2,
                 last_activation='sigmoid'):
        super(FastText, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.avg_pooling = GlobalAveragePooling1D()
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of FastText must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of FastText must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        embedding = self.embedding(inputs)
        x = self.avg_pooling(embedding)
        output = self.classifier(x)
        return output


class TextCNN(Model):

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 kernel_sizes=[3, 4, 5],
                 class_num=2,
                 last_activation='sigmoid'):
        super(TextCNN, self).__init__()
        self.maxlen = maxlen                  # 문장에서 최대 단어 개수 -> 32
        self.max_features = max_features      # 총 단어의 개수 -> ?
        self.embedding_dims = embedding_dims  # word2vec 벡터 크기 -> 300
        self.kernel_sizes = kernel_sizes      # [3, 4, 5]
        self.class_num = class_num            # 2
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        # self.embedding = None
        self.convs = []
        self.max_poolings = []
        for kernel_size in self.kernel_sizes:
            self.convs.append(Conv1D(128, kernel_size, activation='relu'))
            self.max_poolings.append(GlobalMaxPooling1D())
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2: # 원래 2
            raise ValueError('The rank of inputs of TextCNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextCNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        # Embedding part can try multichannel as same as origin paper
        embedding = self.embedding(inputs)
        convs = []
        for i in range(len(self.kernel_sizes)):
            c = self.convs[i](embedding)
            # c = self.convs[i](input)
            c = self.max_poolings[i](c)
            convs.append(c)
        x = Concatenate()(convs)
        output = self.classifier(x)
        return output
    
    
class RCNNVariant(Model):
    """Variant of RCNN.
        Base on structure of RCNN, we do some improvement:
        1. Ignore the shift for left/right context.
        2. Use Bidirectional LSTM/GRU to encode context.
        3. Use Multi-CNN to represent the semantic vectors.
        4. Use ReLU instead of Tanh.
        5. Use both AveragePooling and MaxPooling.
    """

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 kernel_sizes=[1, 2, 3, 4, 5],
                 class_num=2,
                 last_activation='sigmoid'):
        super(RCNNVariant, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        self.bi_rnn = Bidirectional(LSTM(128, return_sequences=True))
        self.concatenate = Concatenate()
        self.convs = []
        for kernel_size in self.kernel_sizes:
            conv = Conv1D(128, kernel_size, activation='relu')
            self.convs.append(conv)
        self.avg_pooling = GlobalAveragePooling1D()
        self.max_pooling = GlobalMaxPooling1D()
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        embedding = self.embedding(inputs)
        x_context = self.bi_rnn(embedding)
        x = self.concatenate([embedding, x_context])
        convs = []
        for i in range(len(self.kernel_sizes)):
            conv = self.convs[i](x)
            convs.append(conv)
        poolings = [self.avg_pooling(conv) for conv in convs] + [self.max_pooling(conv) for conv in convs]
        x = self.concatenate(poolings)
        output = self.classifier(x)
        return output
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import os


print('Loading data...')
x_train = np.load('x_train.npy', allow_pickle=True)
x_test = np.load('x_test.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)

print('Build model...')
max_features = len(word_to_index) + 1 # 8895
maxlen = max_word_num_in_sent # 26
batch_size = 32
embedding_dims = 300
epochs = 3
patience = 3
# TextCNN, FastText, RCNNVariant
model = FastText(maxlen, max_features, embedding_dims)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

try:
    os.mkdir('./model_checkpoint')
except:
    pass
# output_path = os.getcwd()
output_path = 'model_checkpoint'
ckpt_path = os.path.join(output_path, '{epoch:02d}-{val_accuracy:.4f}.h5')

print('Train...')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, mode='max')
mcp_save = ModelCheckpoint(filepath=ckpt_path,
                           monitor='val_accuracy',
                           save_best_only=True, 
                           # save_weights_only=True,
                           verbose=1)
history = model.fit(x_train_pad, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[early_stopping, mcp_save],
                    validation_data=(x_test_pad, y_test))
import matplotlib.pyplot as plt


def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])

display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)
display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
from keras.models import load_model, model_from_json
import csv


print('Loading data...')
x_val = np.load('x_val.npy', allow_pickle=True)
val_id = np.load('val_id.npy', allow_pickle=True)

print("Loading model...")
# model = load_model("model_checkpoint/03-0.7958.h5")

print('Test...')
result = model.predict(x_val)
print(result.shape)
count_false = 0
count_true = 1
prediction = []
for idx, y_hat in zip(val_id, result):
    label = y_hat.argmax()
    # print(y_hat, label)
    if label == 0:
        count_false += 1
    else:
        count_true += 1
    prediction.append((idx, label))
print(f'predict true: {count_true}, predict false: {count_false}')

print('Save submission...')
with open('submission.csv', 'w', newline='') as write_file:
    csv_writer = csv.writer(write_file)
    csv_writer.writerow(['id', 'target'])
    for val_id, y_hat in prediction:
        csv_writer.writerow([val_id, y_hat])
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))