import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import json
from pandas.io.json import json_normalize
# read data files
raw_data=pd.read_json("../input/datamininglab2/tweets_DM.json",lines=True)
tweets=json_normalize(data=raw_data['_source'])
identify=pd.read_csv("../input/datamininglab2/data_identification.csv")
emotion=pd.read_csv("../input/datamininglab2/emotion.csv")
# rename column names
tweets=tweets.rename(index=str,columns={"tweet.text":"text", "tweet.tweet_id":"tweet_id",
                                       "tweet.hashtags":"hashtags"})
# add identify tags to dataframe
tweets=pd.merge(tweets,identify, on="tweet_id")

#get training set and test set
train_df=tweets[tweets["identification"] == "train"]
test_df=tweets[tweets["identification"] == "test"]

#add emotion column
train_df=pd.merge(train_df,emotion, on="tweet_id")
test_df["emotion"]=""

#drop identification tags
train_df.drop(columns=["identification"],inplace=True)
test_df.drop(columns=["identification"],inplace=True)

#use tweet_id as index
train_df.set_index("tweet_id",inplace=True)
test_df.set_index("tweet_id",inplace=True)
# save to pickle file
train_df.to_pickle("train_df.pkl")
test_df.to_pickle("test_df.pkl")
## load a pickle file
train_df = pd.read_pickle("../input/dm-competition-tweets-emotion/train_df.pkl")
test_df = pd.read_pickle("../input/dm-competition-tweets-emotion/test_df.pkl")
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
s1 = '@remy: This is waaaaayyyy too much for you!!!!!!'
tknzr.tokenize(s1)
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(preserve_case=False)
tfidf = TfidfVectorizer(max_features=20000, stop_words='english',
                                     tokenizer=tknzr.tokenize)

# fitting
tfidf.fit(train_df['text'])
# transforming training sets
X_train = tfidf.transform(train_df['text'])
X_train.shape
# transforming testing sets
X_test = tfidf.transform(test_df['text'])
X_test.shape
# set pointers
y_train = train_df['emotion']
y_test = test_df['emotion']
import pandas as pd
model_compare=pd.read_csv("../input/dm-competition-tweets-emotion/final.csv")
model_compare
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=6,n_jobs=-1,max_iter=1000)
lr.fit(X_train,y_train)
pred_result_lr = lr.predict(X_test)
pred_result_lr.shape
# save the result
test_df['emotion']=pred_result_lr
test_df.drop(columns=['hashtags','text'],inplace=True)
test_df.index.rename('id',inplace=True)
test_df.columns=['emotion']
test_df.to_csv('lr_tfidf.csv')
## load a pickle file
train_df = pd.read_pickle("../input/dm-competition-tweets-emotion/train_df.pkl")
test_df = pd.read_pickle("../input/dm-competition-tweets-emotion/test_df.pkl")
## 使用Tokenizer对词组进行编码
## 当我们创建了一个Tokenizer对象后，使用该对象的fit_on_texts()函数，以空格去识别每个词,
## 可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。
max_words = 20000
max_len = 300
tok = Tokenizer(num_words=max_words)  ## 使用的最大词语数为20000
tok.fit_on_texts(train_df['text'])
## 对每个词编码之后，每句新闻中的每个词就可以用对应的编码表示，即每条新闻可以转变成一个向量了：
train_seq = tok.texts_to_sequences(train_df['text'])
test_seq = tok.texts_to_sequences(test_df['text'])

## 将每个序列调整为相同的长度
train_seq_mat = sequence.pad_sequences(train_seq,maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq,maxlen=max_len)

print(train_seq_mat.shape)
print(test_seq_mat.shape)
## deal with label (string -> one-hot)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
print('check label: ', label_encoder.classes_)
print('\n## Before convert')
print('y_train[0:4]:\n', y_train[0:4])
print('\ny_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)

def label_encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)

y_train = label_encode(label_encoder, y_train)
y_test = label_encode(label_encoder, y_test)

print('\n\n## After convert')
print('y_train[0:4]:\n', y_train[0:4])
print('\ny_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)
# I/O check
input_shape = X_train.shape[1]
print('input_shape: ', input_shape)

output_shape = len(label_encoder.classes_)
print('output_shape: ', output_shape)
## 定义LSTM模型
inputs = Input(name='inputs',shape=[max_len])
## Embedding(词汇表大小,batch大小,每个新闻的词长)
layer = Embedding(max_words+1,128,input_length=max_len)(inputs)
layer = LSTM(128)(layer)
layer = Dense(128,activation="relu",name="FC1")(layer)
layer = Dropout(0.5)(layer)
layer = Dense(output_shape,activation="softmax",name="FC2")(layer)
model = Model(inputs=inputs,outputs=layer)
model.summary()
model.compile(loss="categorical_crossentropy",optimizer=RMSprop(),metrics=["accuracy"])
model_fit = model.fit(train_seq_mat,y_train,batch_size=128,epochs=3,
                      callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)] )
## 当val-loss不再提升时停止训练
# predict the result using our model
pred_result_lstm = label_decode(label_encoder, model.predict(test_seq_mat, batch_size=128))
pred_result_lstm[:5]
# save the result
test_df['emotion']=pred_result_lstm
test_df.drop(columns=['hashtags','text'],inplace=True)
test_df.index.rename('id',inplace=True)
test_df.columns=['emotion']
test_df.to_csv('keras_tfidf.csv')
model_compare=pd.read_csv("../input/dm-competition-tweets-emotion/final.csv")
model_compare