import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,LSTM ,Embedding , Dropout
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize , sent_tokenize
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
df = pd.read_csv('../input/traindata/train.csv')
df
df.dropna(inplace = True)
df.reset_index(inplace =True)
df = df.sample(frac = 1).reset_index(drop = True)
df = df.head(8000)
x = df[['title' , 'author' , 'text']]
y = df['label']
x
x.shape , y.shape
tensorflow.__version__
#preprocessing
import time
s = time.time()
corpus = []
for i in range(len(x)):
    if i+1 % 100 == 0:
        print(i)
    
    text = re.sub('[^a-zA-Z]' , " ", x['text'][i])
    text = text.lower()
    text = text.split()
    
    word = [words for words in text if words not in stopwords.words('english') ]
    word = " ".join(word) 
    corpus.append(word)
print('done')
print((time.time() - s)*1000)
corpus
#one hot
voc_size =6000
one_hot_sentence = [one_hot(words , voc_size) for words in corpus]
one_hot_sentence[:20]
# to make fixed length
max_length_of_sent = 50
embedding_sent = pad_sequences(one_hot_sentence,padding='pre' , maxlen=max_length_of_sent)
embedding_sent[0]
len(embedding_sent[0])
len(embedding_sent)
len(x) , len(y)
embedding_feature_size = 256
#after taking input , how much should be length of feature vector after passing into model
#make model
model = Sequential()
model.add(Embedding(voc_size , embedding_feature_size , input_length = max_length_of_sent ))
model.add(LSTM(256 , return_sequences = True))
model.add(Dropout(0.4))
model.add(LSTM(128))
model.add(Dropout(0.4))
model.add(Dense(1 , activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy',metrics = ['accuracy'],optimizer = 'adam')
model.summary()
X = np.array(embedding_sent)
Y = np.array(y)
X.shape , y.shape
x_train , x_test , y_train , y_test = train_test_split(X ,Y,test_size = 0.25 , random_state =100)
#train
history = model.fit(x_train , y_train , validation_data = (x_test , y_test) , epochs =10 , batch_size= 128)
plt.plot(history.history['accuracy'] , label = 'train_acc')
plt.plot(history.history['val_accuracy'] , label = 'val_acc')
plt.legend()


plt.plot(history.history['loss'] , label = 'train_loss')
plt.plot(history.history['val_loss'] , label ='val_loss')
plt.legend()

y_pred = model.predict_classes(x_test)
y_pred
confusion_matrix(y_pred , y_test)
accuracy_score(y_pred, y_test)
user_inp = 'There were many forest fire and Trump was seen dancing there :)'
#preprocessing
import time
s = time.time()
corpus = []

text = re.sub('[^a-zA-Z]' , " ", user_inp)
text = text.lower()
text = text.split()

word = [words for words in text if words not in stopwords.words('english') ]
word = " ".join(word) 
corpus.append(word)
print('done')
print((time.time() - s)*1000)
user_one_hot = [one_hot(words , voc_size) for words in corpus]
user_one_hot
# to make fixed length
max_length_of_sent = 50
embedding_sent_user = pad_sequences(user_one_hot,padding='pre' , maxlen=max_length_of_sent)
embedding_sent_user
check = np.array(embedding_sent_user)
model.predict_classes(check)
