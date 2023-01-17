
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

path = '../input/movie-review-sentiment-analysis-kernels-only/train.tsv'
features = ['pid','sid','p','s']
sms = pd.read_csv(path,names=features,sep='\t',header=0)
print(sms.shape)
print(sms.head(10))

import matplotlib.pyplot as plt
def plot_bars(auto_prices, cols):
    for col in cols:
        fig = plt.figure(figsize=(6,6)) # define plot area
        ax = fig.gca() # define axis    
        counts = auto_prices[col].value_counts() # find the counts for each unique category
        counts.plot.bar(ax = ax, color = 'blue') # Use the plot.bar method on the counts data frame
        ax.set_title('Number sentiments' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel('freq')# Set text for y axis
        plt.show()

plot_cols = ['s']
plot_bars(sms, plot_cols)  
print(sms.s.value_counts())
X=sms.p
Y=sms.s
import nltk
from nltk.stem import PorterStemmer
ps=PorterStemmer()
l2=[]
review=[]
s2=''
for row in X:
    for words in nltk.word_tokenize(row):
            #print(words)
            l2.append(words.lower())
            l2.append(' ')
    s2=''.join(l2)
    review.append(s2)
    s2=''
    l2=[]
X=review
print(X[:1])
from sklearn.cross_validation import train_test_split
X_train, X_inter, Y_train, Y_inter = train_test_split(X, Y,test_size=0.3,random_state=1)
X_val, X_test, Y_val, Y_test = train_test_split(X_inter, Y_inter,test_size=0.5,random_state=1)
print(len(X_train))
print(len(X_val))
print(len(X_test))
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

max_sentence=len(max(X,key=len))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

from numpy import asarray
embeddings_index = dict()
f = open('../input/glove-t/glove.twitter.27B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
from numpy import zeros
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

max_sentence=len(max(X,key=len))

#tokenizer = Tokenizer()
#tokenizer.fit_on_texts(X_train)
encoded_docs = tokenizer.texts_to_sequences(X_train)
train_x = pad_sequences(encoded_docs, maxlen=max_sentence, padding='post')
print(train_x[0])    

encoded_docs=0
encoded_docs = tokenizer.texts_to_sequences(X_val)
val_x = pad_sequences(encoded_docs, maxlen=max_sentence, padding='post')
print(val_x[1])

encoded_docs=0
encoded_docs = tokenizer.texts_to_sequences(X_test)
test_x = pad_sequences(encoded_docs, maxlen=max_sentence, padding='post')
print(test_x[1])

encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y_train = encoder.transform(Y_train)
dummy_y_train = np_utils.to_categorical(encoded_Y_train)
print(dummy_y_train[:3])

encoded_Y_val = encoder.transform(Y_val)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_val = np_utils.to_categorical(encoded_Y_val)




vocab_size = len(tokenizer.word_index) + 1
from keras.layers import Input, Dense, concatenate, Activation
from keras.models import Model
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding,CuDNNGRU,Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import LocallyConnected1D
from keras.layers import Flatten

"""
tweet_input = Input(shape=(max_sentence,), dtype='int32')

tweet_encoder = Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sentence, trainable=True, weights=[embedding_matrix])(tweet_input)
bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(tweet_encoder)
bigram_branch = GlobalMaxPooling1D()(bigram_branch)
trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(tweet_encoder)
trigram_branch = GlobalMaxPooling1D()(trigram_branch)
fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(tweet_encoder)
fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)

merged = Dense(256, activation='relu')(merged)
#merged = Dropout(0.2)(merged)
merged = Dense(5, activation='softmax')(merged)
model = Model(inputs=[tweet_input], outputs=[merged])
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, dummy_y_train,  validation_data=(val_x, dummy_y_val), epochs=4,batch_size=128,verbose=1)
"""
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding,CuDNNGRU,Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import LocallyConnected1D
from keras.layers import Flatten


model=0
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sentence, trainable=True, weights=[embedding_matrix] ))
model.add(LocallyConnected1D(128, 2,strides=1,padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(256, activation='relu'))          
model.add(Dense(5, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, dummy_y_train,  validation_data=(val_x, dummy_y_val), epochs=4,batch_size=128,verbose=1)

import sklearn.metrics as sklm
predictions=model.predict(test_x)
pred=[]
for idx,val in enumerate(predictions):
    pred.append(np.argmax(val))


print(len(Y_test))
print(len(pred))
print(set(Y_test))
print(set(pred))
metrics = sklm.precision_recall_fscore_support(Y_test, pred)


print('Accuracy  %0.2f' % sklm.accuracy_score(Y_test, pred))
print('           0     1     2     3     4')
print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1]+ '        %6.2f' % metrics[0][2]+ '        %6.2f' % metrics[0][3]+ '        %6.2f' % metrics[0][4])
print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1]+ '        %6.2f' % metrics[1][2]+ '        %6.2f' % metrics[1][3]+ '        %6.2f' % metrics[1][4])
print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1]+ '        %6.2f' % metrics[2][2]+ '        %6.2f' % metrics[2][3]+ '        %6.2f' % metrics[2][4])

Y_test=pd.Series(Y_test)
pred=pd.Series(pred)
pd.crosstab(Y_test, pred, rownames=['True'], colnames=['Predicted'], margins=True)



path = '../input/movie-review-sentiment-analysis-kernels-only/test.tsv'
features = ['PhraseId','sid','p']
test_frame = pd.read_csv(path,names=features,sep='\t',header=0)


#For test dataset

l2=[]
review=[]
s2=''
for row in test_frame['p']:
    for words in nltk.word_tokenize(row):
            #print(words)
            l2.append(words.lower())
            l2.append(' ')
    s2=''.join(l2)
    review.append(s2)
    s2=''
    l2=[]
test_frame['p_stemmed']=review




encoded_docs=0
encoded_docs = tokenizer.texts_to_sequences(test_frame['p_stemmed'])
temp_test = pad_sequences(encoded_docs, maxlen=max_sentence, padding='post')

predictions=model.predict(temp_test)
pred=[]
for idx,val in enumerate(predictions):
    pred.append(np.argmax(val))

test_frame['Sentiment']=pred
test_frame.drop(['p','sid','p_stemmed'],axis=1,inplace=True)
print(test_frame.head(10))
test_frame.to_csv('output.csv',index=False)