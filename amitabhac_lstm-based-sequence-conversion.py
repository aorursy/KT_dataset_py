import numpy as np 

import pandas as pd 

from keras.layers import Embedding

from keras.layers import Dense, Dropout, Activation



from keras.models import Sequential

from keras.layers import Dense, Activation, Masking

from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import KFold



### Make sure to install seq2seq from https://github.com/farizrahman4u/seq2seq

from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
df = pd.read_csv("../input/en_train.csv")
df.head(10)
df['class'].unique()
df = df[df['class'] == 'DIGIT']



df['before'] = [word.lower() for word in df.before]

print (df.head())
chars = set()

for word in df.before:

    #print chars

    for char in list(str(word)):

        chars.add(char)



print ('chars --', chars)      

print ('before len max =', max([len (word) for word in df.before]))



words = set(['__UNK__'])

for sentence in df.after:

    for word in sentence.split():

        words.add(word)



print ('after len max =', max([len(word.split()) for word in df.after]))

print ('words --', words)



print ('df len = ', len(df))



print ('chars len = ', len(chars))



char_indices = dict((c, i) for i, c in enumerate(chars))

indices_char = dict((i, c) for i, c in enumerate(chars))



print ('words  len = ', len(words))





word_indices = dict((w, i) for i, w in enumerate(words))

indices_word = dict((i, w) for i, w in enumerate(words))
input_length = 20

output_length = 20



X = np.zeros((len(df.before), input_length, len(chars)), dtype=np.bool)

y = np.zeros((len(df.before), output_length, len(words)), dtype=np.bool)



i = 0

for index, row in df.iterrows():

    for t, char in enumerate(str(row['before'])[:input_length]):

        X[i, t, char_indices[char]] = 1

    

    

    t = 0

    for word in row['after'].split()[:output_length]:

        

        y[i, t, word_indices[word]] = 1

        t += 1

        

    for t in range (t, output_length):

        y[i, t, word_indices['__UNK__']] = 1

    i += 1



print ('data loaded')
print (X.shape)

print (y.shape)
input_dim = len(chars)

output_dim = len(words)

hidden_dim = 64









model = Sequential()







model.add(AttentionSeq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, 

                           output_length=output_length, 

                           input_shape=(input_length, input_dim)))



model.add(Dense(len(words)))

model.add(Activation('softmax'))

#model.compile(loss='mse', optimizer='adam')

optimizer = RMSprop(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

print (model.summary())
X1 = X[:-200]

y1 = y[:-200]



print (X1.shape)

print (y1.shape)





# checkpoint

filepath="weights.best.digit.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

callbacks_list = [checkpoint, early_stop]



kf = KFold(n_splits=5)



for train_index, test_index in kf.split(X1):

    X_train, X_test = X1[train_index], X1[test_index]

    y_train, y_test = y1[train_index], y1[test_index]

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, callbacks=callbacks_list)

    model.load_weights("weights.best.digit.hdf5")



print ('training done')
X2 = X[-200:]

model.load_weights("weights.best.digit.hdf5")

y2 = model.predict(X2)



#print (y2)
predicted = []

for i in y2:

    out = []

    for j in i:

        if (indices_word[np.argmax(j)] != '__UNK__'):

            out.append(indices_word[np.argmax(j)])

            out.append(" ")

    s =  "".join(out)

    #print s

    predicted.append(s)

    
df['predicted'] = np.zeros(len(df))

df['predicted'][-200:] = predicted

#print df[['before', 'after', 'predicted']][-200:]



count = 0

print ('######### Wrong prediction - ')

for index, row in df[-200:].iterrows():

    #print row

    

    if row['after'].strip() != row['predicted'].strip():

        print (row['before'], 'after = ', row['after'], 'predicted=', row['predicted'])

        count +=1

print ('accuracy on test data - ', 1.0 - count/200.0)