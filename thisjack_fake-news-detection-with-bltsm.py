# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))



import matplotlib.pyplot as plt
# 讀取檔案 (Read Data)

data = pd.read_csv('../input/train.csv')



# 隨機抽樣 比率為100% (Randomly Smaple data, ratio is 100%)

data = data.sample(frac = 1)



# 顯示資料的前五筆 (Show Data)

data.head()
# 新增Length欄位，紀錄每個標題文字總長度

# (Create a new column called Length that record every Headline length)

data['Length'] = [len(headline) for headline in data['title'].fillna('')]

data.head()
detail = data['Length'].describe()

print(detail)
# Get all fake news

fliter = data['label'] == 1

pos = data[fliter]

print('是假新聞的數量(原始)：', len(pos))



# Get all true news

fliter = data['label'] == 0

neg = data[fliter]

print('真新聞的數量(原始)：', len(neg))



the_mean = min(len(pos), len(neg))

print("==============================================")



# 抓取所有 Label為1的 資料，並重新洗牌 (Random Shuffle)

p_data = pos.sample(n = the_mean)

print('取樣是假新聞資料總數：', len(p_data))



# 抓取所有 Label為0的 資料，並重新洗牌 (Random Shuffle)

n_data = neg.sample(n = the_mean)

print('取樣不是假新聞資料總數：', len(n_data))
test_split = 0.2

train_split = 1 - test_split



# 隨機抽樣80%的資料當訓練資料 而剩下的20%則當為測試資料

# (20% for Testing Data, others 80% for Training Data)

p_train_data = p_data.sample(frac = train_split)

p_test_data = p_data.drop(p_train_data.index)



n_train_data = n_data.sample(frac = train_split)

n_test_data = n_data.drop(n_train_data.index)



# 合併兩個類別的訓練資料與測試資料

# (Combined fake news and true news)

train_data = pd.concat([p_train_data, n_train_data])

test_data = pd.concat([p_test_data, n_test_data])



# 全部隨機洗牌 (Random Shuffle)

train_data = train_data.sample(frac = 1)

test_data = test_data.sample(frac = 1)



x_train_data = train_data['title'].fillna('')

y_train_data = train_data['label']

x_test_data = test_data['title'].fillna('')

y_test_data = test_data['label']



print('Train Data的Feature數量(已混合非假與假新聞)：', len(x_train_data))

print('Train Data的Label數量(已混合非假與假新聞)：', len(y_train_data))

print('Test Data的Feature數量(已混合非假與假新聞)：', len(x_test_data))

print('Test Data的Label數量(已混合非假與假新聞)：', len(y_test_data))
from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer
# token字典數量 (最常出現的4000字) (Create a token dictionary)

token_num = 4000 



# 擷取多少固定長度字數 (抓標題字數的平均值 72個長度)

# (Get a fix length, we chose the mean of the Headline length)

data_length = int(detail['mean'])



# 輸入向量維度 (Word Embeding output vector dimension)

output_length = 32 



dropout = 0.5

lstm_dim = 256
token = Tokenizer(num_words = token_num, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

token.fit_on_texts(x_train_data)
x_train_seq = token.texts_to_sequences(x_train_data)

x_test_seq = token.texts_to_sequences(x_test_data)
x_train = sequence.pad_sequences(x_train_seq, maxlen = data_length)

x_test = sequence.pad_sequences(x_test_seq, maxlen = data_length)
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.layers.embeddings import Embedding

from keras.layers.recurrent import LSTM

from keras.layers import Bidirectional, TimeDistributed

from keras.callbacks import EarlyStopping
model = Sequential()

model.add(Embedding(output_dim = output_length, input_dim = token_num, input_length = data_length))

model.add(Dropout(dropout))



# using BLSTM (this will be better than LSTM, Avg acc is around 0.94

model.add(Bidirectional(LSTM(lstm_dim), merge_mode = 'sum'))

model.add(Dropout(dropout))



# using LSTM, Avg acc is around 0.93

# model.add(LSTM(lstm_dim))

# model.add(Dropout(dropout))



model.add(Dense(units = 256, activation = 'relu'))

model.add(Dropout(dropout))



model.add(Dense(units = 1, activation = 'sigmoid'))

print(model.summary())
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# 提早結束

es = EarlyStopping(monitor = 'val_loss', patience = 2, verbose = 2)
train_history = model.fit(x = x_train,

                         y = y_train_data,

                         validation_split = 0.2,

                         epochs = 20,

                         batch_size = 512,

                         verbose = 1,

                         callbacks = [es])
%matplotlib inline

def show_train_history(train_history, train, validation):

    plt.plot(train_history.history[train])

    plt.plot(train_history.history[validation])

    plt.title('Train History')

    plt.ylabel(train)

    plt.xlabel('Epoch')

    plt.legend(['train', 'validation'], loc = 'upper left')

    plt.show()
show_train_history(train_history, 'acc', 'val_acc')

show_train_history(train_history, 'loss', 'val_loss')
scores = model.evaluate(x = x_test, y = y_test_data)

scores[1]
# 讀取檔案

test_data = pd.read_csv('../input/test.csv')



# 顯示資料的前五筆

test_data.head()
test_id = test_data['id']

x_test_data = test_data['title'].fillna("")
x_test_seq = token.texts_to_sequences(x_test_data)

x_test = sequence.pad_sequences(x_test_seq, maxlen = data_length)



predict = model.predict_classes(x_test)

predict_classes = predict.reshape(-1)
test_data['label'] = [predict for predict in predict_classes]

test_data.head()
result = test_data[['id', 'label']]

result.head()
# Any results you write to the current directory are saved as output.

result.to_csv('submission.csv', index = False)