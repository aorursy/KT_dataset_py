import numpy as np 

import pandas as pd 



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC



from keras.models import Sequential

from keras.preprocessing.text import Tokenizer

from keras import layers

from keras.layers import Embedding,LSTM

from keras.preprocessing.sequence import pad_sequences



import os

print(os.listdir("../input"))

#reading the three data files and adding the source column

df_yelp=pd.read_csv('../input/sentiment labelled sentences/sentiment labelled sentences/yelp_labelled.txt', names=['sentence', 'label'], sep='\t')

df_imdb=pd.read_csv('../input/sentiment labelled sentences/sentiment labelled sentences/imdb_labelled.txt', names=['sentence', 'label'], sep='\t')

df_amazon=pd.read_csv('../input/sentiment labelled sentences/sentiment labelled sentences/amazon_cells_labelled.txt', names=['sentence', 'label'], sep='\t')



df_yelp['source']='Yelp'

df_imdb['source']='imdb'

df_amazon['source']='Amazon'
#concating the three files to one dataframe

df=pd.concat([df_yelp,df_imdb,df_amazon],ignore_index=True)

df
df.shape
# split the data into test and train split

X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.30, random_state=100)



cnt_vector = CountVectorizer(stop_words='english')

cnt_vector.fit(X_train)

X_train_cnt = cnt_vector.transform(X_train)

X_test_cnt  = cnt_vector.transform(X_test)
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transform=TfidfTransformer()

X_train_tfidf=tfidf_transform.fit_transform(X_train_cnt)

X_test_tfidf=tfidf_transform.fit_transform(X_test_cnt)



# fitting a logistic regression model on the input feature vector

lr = LogisticRegression()

lr.fit(X_train_tfidf, y_train)

y_pred=lr.predict(X_test_tfidf)

from sklearn.metrics import accuracy_score

print('Accuracy: %.4f' % accuracy_score(y_test,y_pred))

#calculating accuracy for each source

for source in df['source'].unique():

    df_source = df[df['source'] == source]

    X = df_source['sentence'].values

    y = df_source['label'].values



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=100)



    cnt_vector = CountVectorizer()

    cnt_vector.fit(X_train)

    X_train_cnt = cnt_vector.transform(X_train)

    X_test_cnt  = cnt_vector.transform(X_test)

    

    tfidf_transform=TfidfTransformer()

    X_train_tfidf=tfidf_transform.fit_transform(X_train_cnt)

    X_test_tfidf=tfidf_transform.fit_transform(X_test_cnt)



    lr = LogisticRegression()

    lr.fit(X_train_tfidf, y_train)

    y_pred=lr.predict(X_test_tfidf)

 

    print('Accuracy for {} data: {:.4f}'.format(source, accuracy_score(y_test,y_pred)))


nb=MultinomialNB()

nb.fit(X_train_tfidf, y_train)

y_pred_nb=nb.predict(X_test_tfidf)

print('Accuracy: %.4f' % accuracy_score(y_test,y_pred_nb))


svc=SVC(kernel='linear')

svc.fit(X_train_tfidf, y_train)

y_pred_svc=svc.predict(X_test_tfidf)

print('Accuracy: %.4f' % accuracy_score(y_test,y_pred_svc))
input_dim = X_train_tfidf.shape[1]

model = Sequential()

model.add(layers.Dense(100, input_dim=input_dim, activation='relu'))

model.add(layers.Dense(1, activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.summary()
model.fit(X_train_tfidf, y_train, epochs=10,verbose=True,validation_data=(X_test_tfidf, y_test), batch_size=10)
loss, accuracy = model.evaluate(X_test_tfidf, y_test, verbose=False)

print("Test Accuracy:  {:.4f}".format(accuracy))

print("Test loss: {:.4f}".format(loss))



token = Tokenizer(num_words=10000)

token.fit_on_texts(X_train) # assign index value to the words



seq_train = token.texts_to_sequences(X_train) # return array of indexes for each sentences

seq_test = token.texts_to_sequences(X_test)



vocab_size = len(token.word_index) + 1 





# padding sequences



padded_train = pad_sequences(seq_train, padding='post', maxlen=100)

padded_test = pad_sequences(seq_test, padding='post', maxlen=100)



# building sequential model using keras embedding layer



model = Sequential()

model.add(layers.Embedding(input_dim=vocab_size, output_dim=100, input_length=100))

#model.add(LSTM(256))

model.add(layers.Flatten())

model.add(layers.Dense(100, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

model.summary()
model.fit(padded_train, y_train,epochs=20, verbose=False,validation_data=(padded_test, y_test),batch_size=10)

loss, accuracy = model.evaluate(padded_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(padded_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

#calculating accuracy for each source

for source in df['source'].unique():

    df_source = df[df['source'] == source]

    X = df_source['sentence'].values

    y = df_source['label'].values

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=100)



    token = Tokenizer(num_words=10000)

    token.fit_on_texts(X_train) # assign index value to the words

    

    seq_train = token.texts_to_sequences(X_train) # return array of indexes for each sentences

    seq_test = token.texts_to_sequences(X_test)

    

    vocab_size = len(token.word_index) + 1 

    

    padded_train = pad_sequences(seq_train, padding='post', maxlen=100)

    padded_test = pad_sequences(seq_test, padding='post', maxlen=100)

    

    model = Sequential()

    model.add(layers.Embedding(input_dim=vocab_size, output_dim=100, input_length=100))

    #model.add(LSTM(256))

    model.add(layers.Flatten())

    model.add(layers.Dense(100, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

    model.summary()

    model.fit(padded_train, y_train,epochs=20, verbose=False,validation_data=(padded_test, y_test),batch_size=10)

    

    loss, accuracy = model.evaluate(padded_test, y_test, verbose=False)

    print('Testing Accuracy for {} data: {:.4f}'.format(source,accuracy))    