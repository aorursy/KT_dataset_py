import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.layers import Merge,merge,concatenate
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,Lambda
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop
from keras import backend as K
x1 = np.load(open('../input/preprocessing-final/q1_train.npy', 'rb'))
x2 = np.load(open('../input/preprocessing-final/q2_train.npy', 'rb'))
y_train = np.load(open('../input/preprocessing-final/y_train.npy', 'rb'))
embedding_matrix = np.load(open('../input/preprocessing-final/embedding_matrix.npy', 'rb'))
word_index = np.load(open('../input/preprocessing-final/word_index.npy', 'rb'))
word_index
def vec_distance(vects):
    x, y = vects
    return K.sum(K.square(x - y), axis=1, keepdims=True)
def vec_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
nb_words=137043
word_embedding_dim=300
sequence_length=25
max_len=25
w1 = Input(shape=(sequence_length,), dtype='int32')
w2 = Input(shape=(sequence_length,), dtype='int32')
#Representation Layer 
embedding_layer = Embedding(nb_words,word_embedding_dim,
        weights=[embedding_matrix],
        input_length=max_len,trainable=True)
sequence1 = embedding_layer(w1)
sequence2 = embedding_layer(w2)

#Context Layer
lstm_layer =LSTM(128)

context1= lstm_layer(sequence1)
context2 =lstm_layer(sequence2)

distance=Lambda(vec_distance, output_shape=vec_output_shape)([context1, context2])
product = merge([context1,context2], mode= "mul")
product=Dropout(0.4)(product)
merged = concatenate([distance, product])
dense1=Dense(16, activation='sigmoid')(merged)
dense1 = Dropout(0.3)(dense1)

bn2 = BatchNormalization()(dense1)
prediction=Dense(1, activation='sigmoid')(bn2)
model = Model(inputs=[w1, w2], outputs=prediction)
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
early_stopping =EarlyStopping(monitor='val_loss', patience=3)
history=model.fit([x1,x2], y_train,validation_split=0.1, verbose=1, 
          epochs=15, batch_size=256, shuffle=True,class_weight=None, callbacks=[early_stopping])
f = open('history.txt','w')
f.write(history.history['acc'][-1].astype(str)+" "+history.history['val_acc'][-1].astype(str))
f.close()

import pickle
with open('trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

# export model to JSON
from keras.models import model_from_json
model_json = model.to_json()
with open("LSTM_model_quora.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("LSTM_model_quora.h5")
print("Model Saved")
# load json and create model
json_file = open('LSTM_model_quora.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("LSTM_model_quora.h5")
print("Loaded model from disk")
q1_test = np.load(open('../input/preprocessing-final/q1_test.npy', 'rb'))
q2_test = np.load(open('../input/preprocessing-final/q2_test.npy', 'rb'))
prediction=model.predict([q1_test,q2_test],verbose=0)
print("Writing output...")
sub = pd.DataFrame()
data_test=pd.read_csv('../input/quora-question-pairs/test.csv')
sub['test_id'] = data_test['test_id']
sub['is_duplicate'] =prediction
sub.to_csv("submission.csv", index=False)


