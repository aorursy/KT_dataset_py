import numpy as np
import pandas as pd
import keras
import pickle

from collections import Counter
nonalphanums = ''.join(c for c in map(chr, range(256)) if not c.isalnum())
nonalphanums = nonalphanums.translate(str.maketrans('','',' '))
raw = pd.read_csv("../input/train.csv")
questiondata = np.array(raw.iloc[:,1])
y = np.array(raw.iloc[:,2])
appendation = " ".join(questiondata).lower()
translator = str.maketrans('', '', nonalphanums)
appendation = appendation.translate(translator)
wordcollection = appendation.split(" ")
occ = dict(Counter(wordcollection))
wordocc =sorted(occ, key=occ.get, reverse=True)
topnumber = 1001
topocc = (wordocc[:topnumber-1])
print(topocc)
topocc.insert(0,"<pad>")
print(topocc)
def dechunker(chunked):
    return np.array(chunked.split(" "))
dechunked = np.array(list(map(dechunker,questiondata)))
def cleanlink(assortment):
    returns = []
    for i in assortment:
        cleansed = i.lower().translate(translator)
        if (cleansed in topocc):
            returns.append(topocc.index(cleansed))
    return np.array(returns)
cleanlinked = np.array(list(map(cleanlink,dechunked)))
longestlinked = max(list(map(len,cleanlinked)))
X = keras.preprocessing.sequence.pad_sequences(cleanlinked,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=longestlinked)
Y = y

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout,LSTM
x,y = X,Y
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2)
epochs = 30
bsize = 100
model = Sequential()
model.add(Embedding(topnumber, 16))
model.add(LSTM(16))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epochs, batch_size=bsize, validation_data=(x_test, y_test), verbose=1)
nonalphanums = ''.join(c for c in map(chr, range(256)) if not c.isalnum())
nonalphanums = nonalphanums.translate(str.maketrans('','',' '))
raw = pd.read_csv("../input/test.csv")
questiondata = np.array(raw.iloc[:,1])
y = np.array(raw.iloc[:,0])
appendation = " ".join(questiondata).lower()
translator = str.maketrans('', '', nonalphanums)
appendation = appendation.translate(translator)
wordcollection = appendation.split(" ")
occ = dict(Counter(wordcollection))
wordocc =sorted(occ, key=occ.get, reverse=True)
topnumber = 1000
topocc = (wordocc[:topnumber])
print(topocc)
topocc.insert(0,"<pad>")
print(topocc)
def dechunker(chunked):
    return np.array(chunked.split(" "))
dechunked = np.array(list(map(dechunker,questiondata)))
def cleanlink(assortment):
    returns = []
    for i in assortment:
        cleansed = i.lower().translate(translator)
        if (cleansed in topocc):
            returns.append(topocc.index(cleansed))
    return np.array(returns)
cleanlinked = np.array(list(map(cleanlink,dechunked)))
longestlinked = max(list(map(len,cleanlinked)))
x = keras.preprocessing.sequence.pad_sequences(cleanlinked,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=longestlinked)
test_x = x
ids = y


results = model.predict(test_x)
#package for submission
import csv
final = []
g = ['qid','prediction']
for it in range(0,len(results)):
  tx = int(not (results[it]>np.mean(results)))
  appendition = [ids[it],tx]
  final.append(appendition)
final = np.array(final)
print(final)
ar = final
with open('submission.csv', 'w+') as fp:
    writer = csv.writer(fp, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(i for i in g)
    writer.writerows(ar.tolist())