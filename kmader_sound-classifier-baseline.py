batch_size = 32
BASE_FILTER_COUNT = 16
max_steps = 10
SAMPLING_RATE = 8000 # [4000, 8000, 16000, 22000]
input_length = SAMPLING_RATE*2
%matplotlib inline  
import numpy as np
import pandas as pd
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from numpy import random
import librosa
import numpy as np
import glob
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+0.0001)
    return data-0.5

def load_audio_file(file_path, input_length=input_length):
    data = librosa.core.load(file_path, sr=SAMPLING_RATE)[0] #, sr=16000
    if len(data)>input_length:
        max_offset = len(data)-input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length+offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
    data = audio_norm(data)
    return data
train_files = glob.glob("../input/audio_train/audio_train/*.wav")
test_files = glob.glob("../input/audio_test/audio_test/*.wav")
train_labels = pd.read_csv("../input/train.csv")
print(len(train_files), 'training', len(test_files), 'testing')
train_labels.groupby(['label']).size().plot.bar()
train_labels.sample(3)
file_to_label = {"../input/audio_train/audio_train/{}".format(k):v 
                 for k,v in zip(train_labels['fname'].values,
                                train_labels['label'].values)}
data_base = load_audio_file(train_files[0])
fig = plt.figure(figsize=(14, 8))
plt.title('Raw wave : %s ' % (file_to_label[train_files[0]]))
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
plt.plot(np.linspace(0, input_length/SAMPLING_RATE, input_length), data_base)
plt.show()
list_labels = sorted(list(set(train_labels['label'].values)))
label_to_int = {k:v for v,k in enumerate(list_labels)}
int_to_label = {v:k for k,v in label_to_int.items()}
file_to_int = {k:label_to_int[v] for k,v in file_to_label.items()}
from keras import layers
def create_model(n_filt, act_name = 'relu'):
    if act_name=='relu':
        act_fun = activations.relu
    elif act_name=='leakyrelu':
        act_fun = layers.LeakyReLU(0.3)
    nclass = len(list_labels)
    inp = Input(shape=(input_length, 1))
    img_1 = Convolution1D(n_filt, kernel_size=9, activation=act_fun, padding="valid")(inp)
    img_1 = Convolution1D(n_filt, kernel_size=9, activation=act_fun, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=16)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(n_filt*2, kernel_size=3, activation=act_fun, padding="valid")(img_1)
    img_1 = Convolution1D(n_filt*2, kernel_size=3, activation=act_fun, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=4)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(n_filt*4, kernel_size=3, activation=act_fun, padding="valid")(img_1)
    img_1 = Convolution1D(n_filt*4, kernel_size=3, activation=act_fun, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=4)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(n_filt*16, kernel_size=3, activation=act_fun, padding="valid")(img_1)
    img_1 = Convolution1D(n_filt*16, kernel_size=3, activation=act_fun, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(n_filt*16, activation=act_fun)(img_1)
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(lr=1e-4)
    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
def train_generator(raw_list_files, batch_size=batch_size):
    while True:
        list_files = np.random.permutation(raw_list_files)
        for batch_files in chunker(list_files, size=batch_size):
            batch_data = [load_audio_file(fpath) for fpath in batch_files]
            batch_data = np.array(batch_data)[:,:,np.newaxis]
            batch_labels = [file_to_int[fpath] for fpath in batch_files]
            batch_labels = np.array(batch_labels)
            yield batch_data, batch_labels
tr_files, val_files = train_test_split(train_files, test_size=0.1, random_state=2018)
%%time
# test the generator
_tx, _ty = next(train_generator(tr_files))
print(_tx.shape, _ty.shape)
model = create_model(BASE_FILTER_COUNT, 'relu')
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
weight_path="{}_weights.best.hdf5".format('simple_sound_model')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
model.fit_generator(train_generator(tr_files), 
                    steps_per_epoch=min(len(tr_files)//batch_size, max_steps), 
                    epochs=50,
                    validation_data=train_generator(val_files), 
                    validation_steps=min(len(val_files)//batch_size, max_steps),
                    callbacks=callbacks_list,
                    use_multiprocessing=True,
                    workers=5)
model.save("baseline_cnn.h5")
list_preds = []
for batch_files in tqdm(chunker(test_files, size=batch_size), total=len(test_files)//batch_size ):
    batch_data = [load_audio_file(fpath) for fpath in batch_files]
    batch_data = np.array(batch_data)[:,:,np.newaxis]
    preds = model.predict(batch_data).tolist()
    list_preds += preds
array_preds = np.array(list_preds)
list_labels = np.array(list_labels)
top_3 = list_labels[np.argsort(-array_preds, axis=1)[:, :3]] #https://www.kaggle.com/inversion/freesound-starter-kernel
pred_labels = [' '.join(list(x)) for x in top_3]
df = pd.DataFrame(test_files, columns=["fname"])
df['label'] = pred_labels
df['fname'] = df.fname.apply(lambda x: x.split("/")[-1])
df.to_csv("baseline.csv", index=False)
