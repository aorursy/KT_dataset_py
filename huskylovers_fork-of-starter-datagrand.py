import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd 

from tqdm import tqdm

from collections import Counter

from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

import seaborn as sns

from gensim.models import Word2Vec

from sklearn.preprocessing import  OneHotEncoder

from sklearn.model_selection import train_test_split

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler

from tqdm import tqdm

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split

print(os.listdir('../input/datagrand/datagrand/'))
with open('../input/datagrand/datagrand/datagrand/train.txt') as f:

    train_num = len(f.readlines())

    for line in f.readlines()[:10]:

        print(line)

with open('../input/datagrand/datagrand/datagrand/test.txt') as f:

    test_num = len(f.readlines())

print("train lines:" + str(train_num))

print("test lines :" + str(test_num))
def micro_f1(sub_lines, ans_lines, split = ' '):

    correct = []

    total_sub = 0

    total_ans = 0

    for sub_line, ans_line in zip(sub_lines, ans_lines):

        sub_line = set(str(sub_line).split(split))

        ans_line = set(str(ans_line).split(split))

        c = sum(1 for i in sub_line if i in ans_line) if sub_line != {''} else 0

        total_sub += len(sub_line) if sub_line != {''} else 0

        total_ans += len(ans_line) if ans_line != {''} else 0

        correct.append(c)

    p = np.sum(correct) / total_sub if total_sub != 0 else 0

    r = np.sum(correct) / total_ans if total_ans != 0 else 0

    f1 = 2*p*r / (p + r) if (p + r) != 0 else 0

    print('total sub:', total_sub)

    print('total ans:', total_ans)

    print('correct: ', np.sum(correct), correct)

    print('precision: ', p)

    print('recall: ',r)

    return 'f1',f1
#train_data 

data = []

targets = []

with open('../input/datagrand/datagrand/datagrand/train.txt') as f:

    for line in f.readlines():

        line_data = []

        target = []

        delimiter = '\t'

        words = line.replace('\n','').split('  ')

        for j,word in enumerate(words):

            split_word = word.split('/')

            tag = split_word[1]

            word_meta = split_word[0]

            word_meta_split = word_meta.split('_')

            line_data += word_meta_split

            meta_len = len(word_meta_split)

            if tag == 'a':

                if meta_len == 1:

                    target.append('B_a')

                else:

                    for k, char in enumerate(word_meta_split):

                        if k == 0:

                            target.append('B_a')

                        elif k == meta_len - 1:

                            target.append('I_a')

                        else:

                            target.append('I_a')

            if tag == 'b':

                if meta_len == 1:

                    target.append('B_b')

                else:

                    for k, char in enumerate(word_meta_split):

                        line_data.append(char)

                        if k == 0:

                            target.append('B_b')

                        elif k == meta_len - 1:

                            target.append('I_b')

                        else:

                            target.append('I_b')

            if tag == 'c':

                if meta_len == 1:

                    target.append('B_c')

                else:

                    for k, char in enumerate(word_meta_split):

                        if k == 0:

                            target.append('B_c')

                        elif k == meta_len - 1:

                            target.append('I_c')

                        else:

                            target.append('I_c')

            else:

                if meta_len == 1:

                    target.append('O')

                else:

                    for k, char in enumerate(word_meta_split):

                        target.append('O')

        targets.append(target)

        data.append(line_data)

#test_data

test_data = []

with open('../input/datagrand/datagrand/datagrand/test.txt') as f:

    for line in f.readlines():

        line = line.replace('\n','')

        test_data.append(line.split('_'))

!pip install git+https://www.github.com/keras-team/keras-contrib.git

!pip install glove_python
#词频和标签编码

word_counts = Counter([word for sentence in data for word in sentence] + [word for sentence in test_data for word in sentence])

vocab = [w for w, f in iter(word_counts.items()) if f>=1]



tag_counts = Counter([tag for sentence_tag in targets for tag in sentence_tag])

tags = [tag for tag, f in iter(tag_counts.items())]
word2idx = dict((w, i) for i, w in enumerate(vocab))

tag2idx = dict((w, i) for i, w in enumerate(tags))



x = [[word2idx.get(word, 1) for word in sentence] for sentence in data]  # set to <unk> (index 1) if not in vocab

x_test = [[word2idx.get(word, 1) for word in sentence] for sentence in test_data]



y_tag = [[tag2idx.get(tag) for tag in sentence_tag] for sentence_tag in targets]

#one_hot

#y_tag = [[np.eye(len(tags))[tag] for tag in sentence_tag] for sentence_tag in y]
x_train, x_val, y_train, y_val = train_test_split(x,y_tag,test_size=0.2,random_state=2019)

len_train = [len(x_) for x_ in x_train]

len_val = [len(x_) for x_ in x_val]

len_test = [len(x_) for x_ in x_test]
maxlen = 200

x_train = pad_sequences(x_train, maxlen)

x_val = pad_sequences(x_val, maxlen)

x_test = pad_sequences(x_test,maxlen)



y_train = pad_sequences(y_train, maxlen, value=-1)

y_val = pad_sequences(y_val, maxlen, value=-1)
y_train = np.expand_dims(y_train, 2)  #模型要求，最外面为一维

y_val = np.expand_dims(y_val, 2)  #模型要求，最外面为一维
os.listdir('../input/')
from glove import Glove

from glove import Corpus
model = Glove.load('../input/glove256-50epoch/glove.model')



word_index_dict= model.dictionary

embedding_index = model.word_vectors

embedding_matrix = np.zeros((len(word2idx),256))

for word,i in tqdm(word2idx.items()):

    try:

        index = word_index_dict[word]

        embedding_matrix[i] = embedding_index[index]

    except KeyError:

        print(i)

        pass


from keras.models import Sequential

from keras.layers import Embedding, Bidirectional, LSTM

from keras_contrib.layers import CRF

import pickle



EMBED_DIM = 256

BiRNN_UNITS = 256





def create_model(train=True):

    model = Sequential()

    model.add(Embedding(len(vocab), weights=[embedding_matrix],output_dim = 256,trainable=True,mask_zero=True))

    model.add(SpatialDropout1D(0.2))

    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))

    #model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))

    crf = CRF(len(tags), sparse_target=True)

    model.add(crf)

    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

    print(model.summary())

    return model
model = create_model()
len(y_train[0])
model.fit(x_train, y_train, epochs=80, batch_size=256, validation_data = (x_val,y_val), verbose=1,shuffle=True)
tag2idx
for i in range(10):

    print('demo'+str(i))

    raw = model.predict(x_train[i])[-len_train[i]:]

    print('predict:',np.array([np.argmax(row) for row in raw]))

    print('tag    :',np.array(y_train[i].reshape(len(y_train[i]))[-len_train[i]:]))
for i in range(10):

    print('demo'+str(i))

    raw = model.predict(x_val[i])[-len_val[i]:]

    print('predict:',np.array([np.argmax(row) for row in raw]))

    print('tag    :',np.array(y_val[i].reshape(len(y_val[i]))[-len_val[i]:]))
# word_vectors.vocab.keys()