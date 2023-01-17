
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import re
data = pd.read_csv('../input/lstm-sentiment-analysis/Sentiment.csv')
data.head()
data["sentiment"].unique()
data.columns
data.shape
data["text"].head()
data = data[["text" , "sentiment"]]
data.head()
#removing 'Nentral' i just need Positive and Ngative
data = data[data.sentiment != 'Neutral']
data['sentiment'].unique()
#converting all words to lowercase , and then remove all special characters
data["text"] = data["text"].apply(lambda x : x.lower())
data["text"] = data["text"].apply(lambda x : re.sub('[^a-zA-Z0-9\s]' , ' ' , x))

data["text"].head()
#remove 'rt'
for idx , row in data.iterrows():
    row[0] = row[0].replace('rt' , '')
    
data['text'].head() 
#the number of Positive and Negative values
print("The number of Positive values = " , data[data.sentiment == "Positive"].size)
print("The number of Negative values = " , data[data['sentiment'] =='Negative'].size)
max_features= 2000
tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_features , split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = keras.preprocessing.sequence.pad_sequences(X)

X.shape
#splitting the data
y = pd.get_dummies(data['sentiment']).values
validation_size = 1500
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.33 , random_state = 42 , shuffle = True)
X_valid , y_valid = X_test[:validation_size] , y_test[:validation_size]
X_test , y_test = X_test[validation_size:] , y_test[validation_size:]
X_train.shape , X_valid.shape , X_test.shape
#building the LSTM model
embed_dim = 128
lstm_out = 196

model = keras.models.Sequential([
    keras.layers.Embedding(max_features , embed_dim , input_length = X.shape[1]),
    keras.layers.SpatialDropout1D(0.3),
    keras.layers.LSTM(lstm_out , dropout = 0.2 , recurrent_dropout = 0.2),
    keras.layers.Dense(2 , activation = 'softmax')
])

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
model.summary()
#fitting the model
batch_size = 32
model.fit(X_train , y_train , batch_size = batch_size , epochs = 10 ,verbose = 2,  validation_data=(X_valid , y_valid))
#evaluate the model
score , accuracy = model.evaluate(X_test , y_test , verbose = 2 , batch_size = batch_size)
print("score : %.2f"%score)
print("accuracy : %.2f"%accuracy)
#test a predicted tweet
twt = ['The life is very good and all peoples are happy']

twt = tokenizer.texts_to_sequences(twt)
twt = keras.preprocessing.sequence.pad_sequences(twt , maxlen= 28 , dtype = 'int32' , value = 0)
print(twt)
sentiment = model.predict(twt , batch_size = 1 , verbose = 2)[0]
#checking positive or negative
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")