import numpy as np  

import pandas as pd  



from keras.layers import Dense,Input,Bidirectional,Conv1D,GRU

from keras.layers import Embedding,GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D

from keras.preprocessing import text, sequence

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.models import Model

from keras.optimizers import Adam, SGD



from sklearn.model_selection import train_test_split 



import spacy
glove_file = '../input/glove840b300dtxt/glove.840B.300d.txt'

train_file = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

test_file = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')



X_train = train_file["comment_text"].str.lower()

X_test = test_file["comment_text"].str.lower()



y_train = train_file[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
max_features=150000

maxlen=200

embed_size=300



tok=text.Tokenizer(num_words=max_features,lower=True)

tok.fit_on_texts(list(X_train)+list(X_test))

X_train=tok.texts_to_sequences(X_train)

X_test=tok.texts_to_sequences(X_test)

x_train=sequence.pad_sequences(X_train,maxlen=maxlen)

x_test=sequence.pad_sequences(X_test,maxlen=maxlen)
embeddings_index = {}

counter = 0

with open(glove_file,encoding='utf8') as f:

    for line in f:

        counter += 1

        if (counter%1000000)==0:

            print(counter)

        values = line.rstrip().rsplit(' ')

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs
nlp = spacy.load('en');



word_index = tok.word_index



num_words = min(max_features, len(word_index) + 1)

embedding_matrix = np.random.randn(num_words, embed_size)/4

kk = 0

moo = 0

for word, i in word_index.items(): 

    if i >= max_features:

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector

        kk += 1

    else: 

        for x in nlp(word,disable=['parser', 'ner']):

            embedding_vector = embeddings_index.get(x.lemma_)

            if embedding_vector is not None: 

                embedding_matrix[i] = embedding_vector 

                kk += 1

                break

model_input = Input(shape=(maxlen, )) 

x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False)(model_input)

x = SpatialDropout1D(0.1)(x)

x = Bidirectional(GRU(200, return_sequences=True,dropout=0.25,recurrent_dropout=0.25,implementation=1))(x)

x = Conv1D(128, kernel_size = 3)(x)   

avg_pool = GlobalAveragePooling1D()(x)

max_pool = GlobalMaxPooling1D()(x)

x = concatenate([avg_pool, max_pool])   

preds = Dense(6, activation="sigmoid")(x)

model = Model(model_input, preds)  

model.summary()

model.compile(loss='binary_crossentropy',optimizer=Adam(lr=2e-4),metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1)
y_pred = model.predict(x_test,batch_size=1024,verbose=1)



submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred

submission.to_csv('submission.csv', index=False)