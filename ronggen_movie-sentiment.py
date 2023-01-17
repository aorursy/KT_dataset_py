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
#样本比例不平衡，增加比例少的样本
from sklearn.utils import resample
train_2 = train[train['Sentiment']==2]
train_1 = train[train['Sentiment']==1]
train_3 = train[train['Sentiment']==3]
train_4 = train[train['Sentiment']==4]
train_5 = train[train['Sentiment']==0]


train_1_sample = resample(train_1,replace=True,n_samples=79000,random_state=123)
train_3_sample = resample(train_3,replace=True,n_samples=79000,random_state=123)
train_4_sample = resample(train_4,replace=True,n_samples=79000,random_state=123)
train_5_sample = resample(train_5,replace=True,n_samples=79000,random_state=123)

df_upsampled = pd.concat([train_2, train_1_sample,train_3_sample,train_4_sample,train_5_sample])

# from sklearn.feature_extraction.text import CountVectorizer

# vec = CountVectorizer(
#     lowercase=True,     # 英文文本全小写
#     analyzer='word', # 逐个字母解析
#     ngram_range=(1,1),  # 1=出现的字母以及每个字母出现的次数，2=出现的连续2个字母，和连续2个字母出现的频次
#     # trump images are now... => 1gram = t,r,u,m,p... 2gram = tr,ru,um,mp...
#     max_features=1000,  # keep the most common 1000 ngrams
#     preprocessor=None
# )

# X=df_upsampled['Phrase']
# Y=df_upsampled['Sentiment']
# X_test=test['Phrase']

# vec.fit(X)
# X=vec.transform(X)
# X_test=vec.transform(X_test)
#对文本进行预处理
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

X=df_upsampled['Phrase']
Y=df_upsampled['Sentiment']
X_test=test['Phrase']
max_features=20000  #最大单词数
max_length=100      #句子最大长度

#设置分词器
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X)
X=tokenizer.texts_to_sequences(X)
X=pad_sequences(X,maxlen=max_length)

# #对测试集数据做同样处理
X_test=tokenizer.texts_to_sequences(X_test)
X_test=pad_sequences(X_test,maxlen=max_length)

# #将y进行One-Hot编码
Y = to_categorical(df_upsampled['Sentiment'].values)
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
NUM_EPOCHS = 2
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(X_val, Y_val))
sub = pd.read_csv('../input/sampleSubmission.csv')

sub['Sentiment'] = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
sub.to_csv('sub_cnn.csv', index=False)
# #用树结构试试
# from sklearn.ensemble import RandomForestClassifier

# classifier = RandomForestClassifier(n_estimators=50,n_jobs=-1)
# classifier.fit(X_train,Y_train)
# classifier.score(X_val, Y_val)
# sub = pd.read_csv('../input/sampleSubmission.csv')

# sub['Sentiment'] = classifier.predict(X_test)
# sub.to_csv('sub_cnn.csv', index=False)