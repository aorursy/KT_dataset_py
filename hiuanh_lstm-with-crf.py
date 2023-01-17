import pandas as pd
import numpy as np

data = pd.read_csv("../input/abcdef/abc.csv")
data = data.fillna(method="ffill")
data.head(10)
sentences=[]
i=0
while i<len(data):
    tmp=[]
    while (data.iloc[i,0])!="." and data.iloc[i,1] !='.':
        tmp.append(data.iloc[i,0])
        i=i+1
    tmp.append('.')
    sentences.append(np.array(tmp))
    i=i+1

tagged_sentences=[]
i=0
while i<len(data):
    tmp=[]
    while (data.iloc[i,0])!="." and data.iloc[i,1] !='.':
        tmp.append(data.iloc[i,1])
        i=i+1
    tmp.append('.')
    tagged_sentences.append(np.array(tmp))
    i=i+1
print(len(tagged_sentences)),
    
print(len(sentences))
from sklearn.model_selection import train_test_split
 
 
train_sentences, test_sentences, train_tags, test_tags = train_test_split(sentences, tagged_sentences, test_size=0.2)
words, tags = set([]), set([])
# Tách word và tag thành từng từ 1 bỏ vào 2 biến trên 
for s in sentences:
    for w in s:
        words.add(w.lower())
 
for ts in tagged_sentences:
    for t in ts:
        tags.add(t)

# Chuyển về số để training, các giá trị chưa có nhãn thì sẽ thay bằng PAD còn trống thì thay bằng 

word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0  # The special value used for padding
word2index['-OOV-'] = 1  # The special value used for OOVs
#print(word2index)
tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0  # The special value used to padding
len(tag2index)
train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []
 
for s in train_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])
 
    train_sentences_X.append(s_int)
 
for s in test_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
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
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from tensorflow.keras import layers
from keras_contrib.layers import CRF
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
#from sklearn_crfsuite import CRF
#from tf_crf_layer.layer import CRF
!pip install crf_layer
inputs = Input(shape=(MAX_LENGTH,))
model = Embedding(input_dim=len(words) + 1, output_dim=128)(inputs)  # 20-dim embedding
model = Bidirectional(LSTM(256, return_sequences=True))(model)  # variational biLSTM
model = TimeDistributed(Dense(len(tag2index)))(model)
model = Activation('softmax')(model)
#outputs = layers.Dense(1, activation="softmax", name="predictions")(model)  # a dense layer as suggested by neuralNer
crf = CRF(len(tag2index),sparse_target=True)
outputs = crf(model)  # output
model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word2index), 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))
crf = CRF(len(tag2index),sparse_target=True)
model.add(crf)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])
 
model.summary()
model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy']) 
 
model.summary()
def to_categoricalsss(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)
cat_train_tags_y = to_categoricalsss(train_tags_y, len(tag2index))
print(cat_train_tags_y[0])
model.fit(train_sentences_X, to_categoricalsss(train_tags_y, len(tag2index)), batch_size=128,verbose=1, epochs=10, validation_split=0.2)
cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
print(cat_train_tags_y[0])
word2index
train_sentences_X[1]
np.array(to_categorical(train_tags_y, len(tag2index))).shape
np.array(train_sentences_X).shape