from keras.layers import Input, LSTM, Dense, merge, Conv1D, MaxPooling1D, Flatten, Embedding, Dropout

from keras.layers import concatenate

from keras.models import Model

import numpy as np

from keras.utils.np_utils import to_categorical

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import pdb

from keras import backend as K

from keras.engine.topology import Layer

import os

import sys

import jieba
import os

os.listdir('../input/sgnsliteraturebigram/sgns.literature.bigram/')
BASE_DIR = '../input/'

input_train = BASE_DIR + 'chinesestsmaster/ChineseSTS-master/ChineseSTS-master/simtrain_to05sts.txt'  #数据集的地址

GLOVE_DIR = BASE_DIR + 'sgnsliteraturebigram/sgns.literature.bigram/sgns.literature.bigram' # 预训练词向量的地址

MAX_SEQUENCE_LENGTH = 50   #句子最长

MAX_NB_WORDS=200000

EMBEDDING_DIM = 300         #词向量维度 

VALIDATION_SPLIT = 0.2      #测试集比例
embeddings_index = {}  #用字典存放词向量



with open(GLOVE_DIR,'r',encoding='utf8') as f:  # 预训练的词向量

    lines=f.readlines()

    print(lines[0])

    print(lines[1])

for line in lines[1:]:

    values = line.split() #用空格分割

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs



print(embeddings_index['咱'])
vocab= embeddings_index.keys()

print('Found %s word vectors.' % len(vocab))

#打开数据集文件，原句

with open(input_train,'r', encoding='utf8') as file:

    input_text = file.readlines()

    

print(input_text[0:1])

print(len(input_text))
labels = []        #存放标签

train_left = []    #存放第一句话

train_right = []   #存放第二句话
# 读入训练集

max=0

for line in input_text:

    line  = line.strip()    #去除前后空格

    tmp = line

    line = line.split('\t') #分解



    if len(line)<5:

        continue

        

    label_id = line[4]

    title_left = line[1].strip()

    title_right = line[3].strip()

   

    #结巴分词

    seg_list_left = jieba.cut(title_left)  #精确分词

    seg_list_right = jieba.cut(title_right) 

    text_left = (' '.join(seg_list_left)).strip()

    text_right = (' '.join(seg_list_right)).strip()



    labels.append(float(label_id)/5)

    train_left.append(text_left)

    train_right.append(text_right)



    

print('Found %s left.' % len(train_left))

print('Found %s right.' % len(train_right))

print('Found %s labels.' % len(labels))
print(train_left[3])

print(train_right[3])

print(labels[3])
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(vocab)        #生成一个词典  

sequences_left = tokenizer.texts_to_sequences(train_left)  #简单理解此函数先做str.split()，再把单词转为下标

sequences_right = tokenizer.texts_to_sequences(train_right) 
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
print(tokenizer.texts_to_sequences(['你 真帅 呀']))
#对齐句子长度，截长补短

data_left = pad_sequences(sequences_left, maxlen=MAX_SEQUENCE_LENGTH)

data_right = pad_sequences(sequences_right, maxlen=MAX_SEQUENCE_LENGTH)

labels = np.array(labels)
#打乱数据集

indices = np.arange(data_left.shape[0])

np.random.shuffle(indices)

data_left = data_left[indices]     #shape?

data_right = data_right[indices]

labels = labels[indices]          #shape?
#把数据集切分为训练集和测试集两部分

nb_validation_samples = int(VALIDATION_SPLIT * data_left.shape[0]) # create val and sp

input_train_left = data_left[:-nb_validation_samples]

input_train_right = data_right[:-nb_validation_samples]

val_left = data_left[-nb_validation_samples:]

val_right = data_right[-nb_validation_samples:]

labels_train = labels[:-nb_validation_samples]

labels_val = labels[-nb_validation_samples:]
print('Preparing embedding matrix.')



nb_words = len(word_index)               #字典大小

embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))  #初始化词向量矩阵



for word, i in word_index.items():  #word_index.items()存放的是 单词和单词ID

    embedding_vector = embeddings_index.get(word) #单词和单词对应的词向量

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector 
#打包一个复用模块



tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,))

#词向量层

embedding_layer = Embedding(nb_words + 1, EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix], trainable=False)(tweet_input) 



conv1 = Conv1D(128, 3, activation='tanh')(embedding_layer)

drop_1 = Dropout(0.2)(conv1)

max_1 = MaxPooling1D(3)(drop_1)

conv2 = Conv1D(128, 3, activation='tanh')(max_1)

drop_2 = Dropout(0.2)(conv2)

max_2 = MaxPooling1D(3)(drop_2)

out_1 = Flatten()(max_1)



model_encode = Model(tweet_input, out_1)  #打包！

tweet_a = Input(shape=(MAX_SEQUENCE_LENGTH,))    

tweet_b = Input(shape=(MAX_SEQUENCE_LENGTH,))



encoded_a = model_encode(tweet_a)

encoded_b = model_encode(tweet_b)



merged_vector = concatenate([encoded_a, encoded_b]) 



dense_1 = Dense(128,activation='relu')(merged_vector)

dense_2 = Dense(128,activation='relu')(dense_1)

dense_3 = Dense(128,activation='relu')(dense_2)

predictions = Dense(1, activation='sigmoid')(dense_3)





model = Model(input=[tweet_a, tweet_b], output=predictions) #整个模型打包！



model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])





# 下面是训练程序

model.fit([input_train_left,input_train_right], labels_train, nb_epoch=10)

json_string = model.to_json()  # json_string = model.get_config()

open('my_model_architecture.json','w').write(json_string)  

model.save_weights('my_model_weights.h5')  



# 下面是训练得到的神经网络进行评估

score = model.evaluate([input_train_left,input_train_right], labels_train, verbose=0) 

print('train score:', score[0]) # 训练集中的loss

print('train accuracy:', score[1]) # 训练集中的准确率

score = model.evaluate([val_left, val_right], labels_val, verbose=0) 

print('Test score:', score[0])#测试集中的loss

print('Test accuracy:', score[1]) #测试集中的准确率
sentence=['咱俩谁跟谁','我们俩谁跟谁呀']



def predict(sentence,model):

    seg_list_left = jieba.cut(sentence[0])   #分词

    seg_list_right = jieba.cut(sentence[1]) 

    text_left = (' '.join(seg_list_left)).strip()

    text_right = (' '.join(seg_list_right)).strip()

    sequences_left = tokenizer.texts_to_sequences([text_left])  #向量化

    sequences_right = tokenizer.texts_to_sequences([text_right])

    print(text_left,text_right)

    print(sequences_left,sequences_right)

    data_left = pad_sequences(sequences_left, maxlen=MAX_SEQUENCE_LENGTH,padding='pre', truncating='post') #padding

    data_right = pad_sequences(sequences_right, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')

    prediction=model.predict([data_left,data_right]) #调用模型

    return prediction 



predict(sentence,model)[0][0]
