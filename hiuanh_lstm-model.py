


import pandas as pd
import numpy as np

data = pd.read_csv("../input/pos-tagging/pos.csv")
data.head(10)
sentences=[]
i=0
while i<len(data):
    tmp=[]
    while (data.iloc[i,0])!="." and data.iloc[i,1] !='.':
        tmp.append(data.iloc[i,0])
        i=i+1
    tmp.append('.')
    sentences.append(tuple(tmp))
    i=i+1

tagged_sentences=[]
i=0
while i<len(data):
    tmp=[]
    while (data.iloc[i,0])!="." and data.iloc[i,1] !='.':
        tmp.append(data.iloc[i,1])
        i=i+1
    tmp.append('.')
    tagged_sentences.append(tuple(tmp))
    i=i+1
print(len(tagged_sentences)),
    
print(len(sentences))
from sklearn.model_selection import train_test_split
 
 
train_sentences, test_sentences, train_tags, test_tags = train_test_split(sentences, tagged_sentences, test_size=0.2)
words, tags = set([]), set([])
 
for s in train_sentences:
    for w in s:
        words.add(w)
 
for ts in train_tags:
    for t in ts:
        tags.add(t)
# Chuyển về số để training, các giá trị chưa có nhãn thì sẽ thay bằng PAD còn trống thì thay bằng 
word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0 
word2index['-OOV-'] = 1
 
tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0 
train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []
 
for s in train_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w])
        except KeyError:
            s_int.append(word2index['-OOV-'])
 
    train_sentences_X.append(s_int)
 
for s in test_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w])
        except KeyError:
            s_int.append(word2index['-OOV-'])
 
    test_sentences_X.append(s_int)
 
for s in train_tags:
    train_tags_y.append([tag2index[t] for t in s])
 
for s in test_tags:
    test_tags_y.append([tag2index[t] for t in s])
 
print(train_sentences_X[0])
print(test_sentences_X[0])
print(train_tags_y[0])
print(test_tags_y[0])
MAX_LENGTH = len(max(train_sentences_X, key=len))
print(MAX_LENGTH)  # 271
 
from keras.preprocessing.sequence import pad_sequences
 
train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')
 
print(train_sentences_X[0])
print(test_sentences_X[0])
print(train_tags_y[0])
print(test_tags_y[0])
 
!pip install git+https://www.github.com/keras-team/keras-contrib.git
len(word2index)
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from keras_contrib.layers import CRF
 
model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word2index), 128))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])
 
model.summary()
def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)
history  = model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=10, validation_split=0.2)
scores = model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")   
scores = model.evaluate(train_sentences_X, to_categorical(train_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")   # acc: 99.09751977804825
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
test_samples = ["But very mystical too. ".split()]
print(test_samples)

test_samples_X = []
for s in test_samples:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])
    test_samples_X.append(s_int)
 
test_samples_X = pad_sequences(test_samples_X, maxlen=MAX_LENGTH, padding='post')
print(test_samples_X)
predictions = model.predict(test_samples_X)
print(predictions, predictions.shape)
# Transform ngược
def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences

print(logits_to_tokens(predictions, {i: t for t, i in tag2index.items()}))

import matplotlib.pyplot as plt
#  Get all the sentences
sentences = sentences
# Plot sentence by lenght
plt.hist([len(s) for s in sentences], bins=50)
plt.title('Token per sentence')
plt.xlabel('Len (number of token)')
plt.ylabel('# samples')
plt.show()