# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
seed = 1
import random
import numpy as np
import pandas as pd
from tensorflow import set_random_seed


random.seed(seed)
np.random.seed(seed)
set_random_seed(seed)
#load data
train = pd.read_csv('../input/train.tsv',  sep="\t")
test = pd.read_csv('../input/test.tsv',  sep="\t")
#观察数据
train.info()
#数据没有空值,观察数据内容
train.head()
#观察样本的大致统计分布
train['Sentiment'].value_counts()
#对文本进行预处理
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

X=train['Phrase']
Y=train['Sentiment']
X_test=test['Phrase']
max_features=20000  #最大单词数
max_length=100      #句子最大长度

#设置分词器
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X)
X=tokenizer.texts_to_sequences(X)
X=pad_sequences(X,maxlen=max_length)

#对测试集数据做同样处理
X_test=tokenizer.texts_to_sequences(X_test)
X_test=pad_sequences(X_test,maxlen=max_length)

#将y进行One-Hot编码
Y = to_categorical(train['Sentiment'].values)
#对数据进行切分
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=seed)
#搭建模型,利用LSTM搭建
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential

model = Sequential()
# Input / Embdedding
model.add(Embedding(max_features, 128, input_length=max_length))
#LSTM
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
BATCH_SIZE = 32
NUM_EPOCHS = 3
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(X_val, Y_val))
sub = pd.read_csv('../input/sampleSubmission.csv')

sub['Sentiment'] = model.predict_classes(X_test, batch_size=BATCH_SIZE, verbose=1)
sub.to_csv('sub_cnn.csv', index=False)