import os

from glob import glob # class별로 나뉘어 있는 train data를 받아오기 위해 glob module 을 사용

import pandas as pd

import numpy as np

import dask.dataframe as dd # 대용량 파일을 읽기위한 패키지

from dask.diagnostics import ProgressBar

import matplotlib.pyplot as plot

import json 



pbar = ProgressBar()

pbar.register()



print(os.listdir("../input"))
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

import keras 

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

base_dir = os.path.join('..','input')

test_path = os.path.join(base_dir,'test_simplified.csv')

train_paths = glob(os.path.join(base_dir, 'train_simplified', '*.csv'))

# class별로 나뉘어있는 train data를 받아오기 위해 glob module 을 사용



test_data = pd.read_csv(test_path)

print("test_data columns: ",test_data.columns)

train_data = pd.read_csv(train_paths[0])

print("train_data columns: ",train_data.columns)

print("draw data type: ", type(train_data['drawing'][0]))
train_data.head()
# drawing data를 [number of points,3] 크기의 tensor로 바꾸어주는 함수

def make_stroke2tensor(raw_strokes):

    # string type의 drawing data를 list로 변환

    stroke_lst = json.loads(raw_strokes)

    # storke_lst = (coord_x list,coor_y list) 들로 구성

    # -> 쌍이 되는 (x,y,stroke id)들로 이루어진 list를 만듦

    stroke_coords = [(x,y,i) for i,(x_lst,y_lst) in enumerate(stroke_lst) 

                  for x,y in zip(x_lst,y_lst)]

    # (x,y,index)들의 리스트인 그림 데이터 하나를 원소로 가지는 리스트로 구성

    stroke_coords = np.stack(stroke_coords)

    

    # 획의 시작과 끝의 정보 저장 

    stroke_coords[:,2] = [1]+np.diff(stroke_coords[:,2]).tolist()

    stroke_coords[:,2] += 1

    

    # return stroke_coords

    

    return pad_sequences(stroke_coords.swapaxes(0, 1), 

                         maxlen=200, 

                         padding='post').swapaxes(0, 1)



train_data.drawing = train_data.drawing.map(make_stroke2tensor)

# 임의의 train data file 을 살펴보니, train_data의 shape은 약 12만개의 데이터가 6개의 정보를 담고 있는 형태이고,

# drawing data는 1000개의 점으로 이루어진 data이다.

print(train_data.shape)

print(train_data.drawing.shape)

print(train_data.drawing[0].shape)
X = np.stack(train_data.drawing,0)

Y = train_data.drawing

Y = np.stack(Y,0)

print(Y)
X.shape

Y.shape
# train_data.drawing # [[drawing data 나열:[x,y,startORend_info]]]
whole_train_data = dd.read_csv(train_paths)
# whole_train_data.count().compute()
a = whole_train_data[['drawing','word']].sample(frac = 0.008)
train_data,valid_data,train_test_data =a. random_split([0.60,0.20,0.20])
train_data.drawing.head()
train_word = train_data['word']

train_drawing = train_data.drawing

train_drawing = train_drawing.map(make_stroke2tensor,meta=('drawing', int))



valid_word = valid_data['word']

valid_drawing = valid_data.drawing

valid_drawing = valid_drawing.map(make_stroke2tensor,meta=('drawing', int))



train_test_word = train_test_data['word']

train_test_drawing = train_test_data.drawing

train_test_drawing = train_test_drawing.map(make_stroke2tensor,meta=('drawing',int))

# word 형태의 category를 one-hot encoding으로 분류하기 위함 

word_encoder = LabelEncoder()

word_encoder.fit(train_word)



train_word = to_categorical(word_encoder.transform(train_word.values))

valid_word = to_categorical(word_encoder.transform(valid_word.values))

train_test_word = to_categorical(word_encoder.transform(train_test_word.values))



train_drawing =np.stack(np.array(train_drawing),0)

valid_drawing =np.stack(np.array(valid_drawing),0)

train_test_drawing =np.stack(np.array(train_test_drawing),0)

print(train_drawing.shape)

print(train_drawing.dtype)

print(train_word.shape)

print(train_word.dtype)
def preds2catids(predictions):

    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])



def top_3_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=3)
learning_rate = 1e-3

num_epochs = 15

batch_size = 2048

num_display = 100
from keras.layers import CuDNNLSTM as LSTM

from keras.layers import BatchNormalization,Conv1D, Dense, Dropout

# keras 순차모델 생성

from keras.metrics import top_k_categorical_accuracy

model = Sequential()
model.add(BatchNormalization(input_shape = (None,)+train_drawing.shape[2:]))
model.add(Conv1D(64,(3,)))

model.add(Dropout(rate = 0.8))

model.add(Conv1D(128,(3,)))

model.add(Dropout(rate = 0.8))

model.add(Conv1D(256,(3,)))

model.add(Dropout(rate = 0.8))
model.add(LSTM(128,return_sequences = True))

model.add(Dropout(rate = 0.8))
model.add(LSTM(256,return_sequences = False))

model.add(Dropout(rate = 0.8))
model.add(Dense(340,activation = 'softmax'))
model.compile(optimizer = 'adam',

             loss = 'categorical_crossentropy',

             metrics = ['categorical_accuracy',top_3_accuracy])

model.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('model')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)





reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, 

                                   verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=5) # probably needs to be more patient, but kaggle time is limited

callbacks_list = [checkpoint, early, reduceLROnPlat]
from IPython.display import clear_output

model.fit(train_drawing, train_word,

                      validation_data = (valid_drawing, valid_word), 

                      batch_size = batch_size,

                      epochs = 50,

                      callbacks = callbacks_list)

clear_output()
model.load_weights(weight_path)

lstm_results = model.evaluate(train_test_drawing, train_test_word, batch_size = 2048)

print('Accuracy: %2.1f%%, Top 3 Accuracy %2.1f%%' % (100*lstm_results[1], 100*lstm_results[2]))
sub_drawing = test_data['drawing'].map(make_stroke2tensor)

sub_drawing = np.stack(sub_drawing.values,0)

sub_pred = model.predict(sub_drawing,verbose = True, batch_size = 2048)
top_3_pred = [word_encoder.classes_[np.argsort(-1*c_pred)[:3]] for c_pred in sub_pred]
top_3_pred = [' '.join([col.replace(' ', '_') for col in row]) for row in top_3_pred]

top_3_pred[:3]
test_data['word'] = top_3_pred

test_data[['key_id', 'word']].to_csv('submission.csv', index=False)
import matplotlib.pyplot as plt
fig, m_axs = plt.subplots(3,3, figsize = (16, 16))

rand_idxs = np.random.choice(range(sub_drawing.shape[0]), size = 9)

for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):

    test_arr = sub_drawing[c_id]

    test_arr = test_arr[test_arr[:,2]>0, :] # only keep valid points

    lab_idx = np.cumsum(test_arr[:,2]-1)

    for i in np.unique(lab_idx):

        c_ax.plot(test_arr[lab_idx==i,0], 

                np.max(test_arr[:,1])-test_arr[lab_idx==i,1], '.-')

    c_ax.axis('off')

    c_ax.set_title(top_3_pred[c_id])