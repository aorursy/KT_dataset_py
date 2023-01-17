import pandas as pd
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
from keras.layers import Bidirectional,LSTM,Dense,Embedding,Dropout,Activation,Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gensim
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv1D, MaxPooling1D
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer,concatenate,BatchNormalization,GlobalMaxPooling1D,GlobalAveragePooling1D

df=pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1',names=["target", "ids", "date", "flag", "user", "text"])
df.head(5)
nltk.download('stopwords')
stop_words=stopwords.words('english')
stemmer=SnowballStemmer('english')
def preprocessing(text):
    pattern="@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    text=re.sub(pattern," ",str(text).lower()).strip()
    tokens=[]
    for token in text.split():
#         tokens.append(token)
        if token not in stop_words:
            tokens.append(stemmer.stem(token))
    return " ".join(tokens)
df.text=df.text.apply(lambda x: preprocessing(x))
df_train,df_test=train_test_split(df,test_size=0.05,random_state=1)
len(df_test)
documents=[t.split() for t in df_train.text]
documents
w2v_model=gensim.models.word2vec.Word2Vec(size=300,window=5,min_count=10,workers=8)
w2v_model.build_vocab(documents)
words = w2v_model.wv.vocab.keys()
vocab_size = len(words)
print("Vocab size", vocab_size)
w2v_model.train(documents, total_examples=len(documents), epochs=32)
w2v_model.most_similar("love")

tokenizer=Tokenizer()
tokenizer.fit_on_texts(df_train.text)

X_train=pad_sequences(tokenizer.texts_to_sequences(df_train.text),maxlen=280)
X_test=pad_sequences(tokenizer.texts_to_sequences(df_test.text),maxlen=280)
X_train.shape
decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]
df_train.target = df_train.target.apply(lambda x: decode_sentiment(x))
df_test.target = df_test.target.apply(lambda x: decode_sentiment(x))

labels=list(df_train.target.unique())
labels
encoder=LabelEncoder()
y_train=encoder.fit_transform(df_train.target).reshape(-1,1)
y_test=encoder.transform(df_test.target).reshape(-1,1)
vocab_size = len(tokenizer.word_index) + 1

y_train.shape

embedding_matrix=np.zeros((vocab_size,300))
for word,i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i]=w2v_model.wv[word]
embedding_layer=Embedding(vocab_size,300,weights=[embedding_matrix],input_length=280,trainable=False)
bs=1024

callbacks=[
    EarlyStopping(
        patience=4,
        monitor='val_accuracy',
    ),
    
    ReduceLROnPlateau(monitor='loss',
                     factor=0.1,
                     patience=2,
                     cooldown=2,
                     verbose=1)
]
lr_decay = (1./0.8 -1)/bs
opt = Adam(lr=1e-3,beta_1=0.9,beta_2=0.999,decay=0.01)

X_input=Input(shape=(X_train.shape[1],))
X=embedding_layer(X_input)
X=Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2))(X)
output=Dense(1,activation='sigmoid')(X)
model=Model(inputs=X_input,outputs=output)
              
model.summary()
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train, epochs=2, batch_size=bs,validation_split=0.05,callbacks=callbacks)

score = model.evaluate(X_test, y_test, batch_size=512)
print()
print("ACCURACY:",score[1])
print("LOSS:",score[0])
X_input=Input(shape=(X_train.shape[1],))
X=embedding_layer(X_input)
X=Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))(X)
X=Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(X)
X=Dense(32,activation='relu')(X)
X=Dense(32,activation='relu')(X)
output=Dense(1,activation='sigmoid')(X)
model2=Model(inputs=X_input,outputs=output)
model2.summary()
model2.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
model2.fit(X_train,y_train, epochs=2, batch_size=bs,validation_split=0.05,callbacks=callbacks)
score = model2.evaluate(X_test, y_test, batch_size=1024)
print()
print("ACCURACY:",score[1])
print("LOSS:",score[0])
X_input=Input(shape=(X_train.shape[1],))
X=embedding_layer(X_input)
X=Conv1D(250,3,padding='valid',activation='relu',strides=1)(X)
X=MaxPooling1D()(X)
X=Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2))(X)
output=Dense(1,activation='sigmoid')(X)
model3=Model(inputs=X_input,outputs=output)
model3.summary()
# opt = Adam(lr=0.005,beta_1=0.9,beta_2=0.999,decay=0.01)
model3.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
model3.fit(X_train,y_train, epochs=2, batch_size=bs,validation_split=0.05,callbacks=callbacks)
score = model3.evaluate(X_test, y_test, batch_size=1024)
print()
print("ACCURACY:",score[1])
print("LOSS:",score[0])
def predict(text):
    
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=280)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score)

    return {"label": label, "score": float(score)}  
predict("I hate the music")
model.save('model.h5')
model2.save('model2.h5')
model3.save('model3.h5')
w2v_model.save('model.w2v')
class AvgWords(Layer):
    def __init__(self,output_dim=300):
        super(AvgWords,self).__init__()
        self.output_dim=output_dim

    def call(self, x):
        axis = K.ndim(x) - 2
        return K.mean(x, axis=axis)

    def compute_output_shape(self,input_shape):
        return (input_shape[0], self.output_dim)
class WordDropout(Layer):
    def __init__(self,rate):
        super(WordDropout,self).__init__()
        self.rate=rate

    def call(self,inputs):
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        words = input_shape[1]
        mask = K.random_uniform((batch_size, words, 1)) >= self.rate
        w_drop = K.cast(mask, 'float32') * inputs
        return w_drop

no_hidden_layers=5
X_input=Input(shape=(X_train.shape[1],))
X1=embedding_layer(X_input)
X1=WordDropout(0.2)(X1)
X1=AvgWords()(X1)
# X1=GlobalAveragePooling1D()(X1)
for i in range(no_hidden_layers):
    X1=Dense(300,activation='tanh')(X1)
    X1=BatchNormalization()(X1)
    X1=Dropout(0.6)(X1)
# X1=Dense(300,activation='relu')(X1)
# X1=Dense(300,activation='relu')(X1)
out=Dense(1,activation='sigmoid')(X1)
model4=Model(inputs=X_input,outputs=out)
model4.summary()
model4.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
model4.fit(X_train,y_train, epochs=20, batch_size=bs,validation_split=0.05,callbacks=callbacks)
score = model4.evaluate(X_test, y_test, batch_size=1024)
print()
print("ACCURACY:",score[1])
print("LOSS:",score[0])
