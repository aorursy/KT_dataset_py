# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras import layers
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, Flatten, LSTM, Dropout, Flatten
from keras.models import Model, load_model
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread, imshow
from tensorflow.keras.applications.mobilenet import preprocess_input
import keras
import cv2
BATCH_SIZE = 128
MAX_TRAIN_EPOCHS = 20
STEPS_PER_EPOCH = 900
NCSVS = 100
CSV_DIR = '../input/doodle-detection-dataprep'
BASE_SIZE = 256
size = 64
word_encoder = LabelEncoder()
categories = [word.split('.')[0] for word in os.listdir(os.path.join("../input/quickdraw-doodle-recognition/train_simplified/"))]
word_encoder.fit(categories)
print('words', len(word_encoder.classes_), '=>', ', '.join([x for x in word_encoder.classes_]))
len(word_encoder.classes_)
def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img

def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(CSV_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = np.zeros((len(df), size, size, 1))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,
                                             time_color=time_color)
                x = preprocess_input(x).astype(np.float32)
                y = to_categorical(word_encoder.transform(df["word"].values),num_classes=340).astype(np.int32)
                yield x, y

def df_to_image_array_xd(df, size, lw=6, time_color=True):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    x = preprocess_input(x).astype(np.float32)
    return x
train_datagen = image_generator_xd(batchsize=BATCH_SIZE, ks=range(NCSVS - 1), size=size)

train_x, train_y = next(train_datagen)

print ('train x shape:{}'.format(train_x.shape))
print ('train y shape:{}'.format(train_y.shape))
print('train_x', train_x.dtype, train_x.min(), train_x.max())
print('train_y', train_y.dtype, train_y.min(), train_y.max())

valid_set = pd.read_csv(os.path.join(CSV_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=10000)
valid_x = df_to_image_array_xd(valid_set, size)
valid_y = to_categorical(word_encoder.transform(valid_set["word"].values),num_classes=340).astype(np.int32)


print ('valid x shape:{}'.format(valid_x.shape))
print ('valid y shape:{}'.format(valid_y.shape))
print('valid_x', valid_x.dtype, valid_x.min(), valid_x.max())
print('valid_y', valid_y.dtype, valid_y.min(), valid_y.max())
fig, m_axs = plt.subplots(4,4, figsize = (8, 8))
rand_idxs = np.random.choice(range(train_x.shape[0]), size = 16, replace=False)
for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
    test_arr = train_x[c_id, :, :, 0]  
    c_ax.imshow(test_arr, cmap=plt.cm.gray)
    c_ax.axis('off')
    c_ax.set_title(word_encoder.classes_[np.argmax(train_y[c_id])])
def doodle(input_shape):
    input_img = Input(input_shape)
    conv0= Conv2D(256, (3, 3), activation='relu', padding='valid')(input_img) 
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
    conv1= Conv2D(128, (3, 3), activation='relu', padding='valid')(pool0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2= Conv2D(64, (3, 3), activation='relu', padding='valid')(pool1) 
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='valid')(pool2) 
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) 
#     conv4 = Conv2D(16, (3, 3), activation='relu', padding='valid')(pool3) 
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    flat = Flatten()(pool3)
    dense1 = Dense(680, activation='relu')(flat)
    dense2 = Dense(len(word_encoder.classes_), activation = 'softmax')(dense1)
    
    model =  Model(inputs = input_img, outputs = dense2, name = 'Doodle_model')    
    return model
model = doodle(input_shape = train_x.shape[1:])
model.summary()
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['categorical_accuracy', top_3_accuracy])

weight_path="model_weights.best.hdf5"

checkpoint = ModelCheckpoint(weight_path, monitor='val_top_3_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True, period=1)

early = EarlyStopping(monitor="val_top_3_accuracy", mode="max", verbose=2,
                      patience=8) # patience is number of epochs with no improvement after which training will be stopped

callbacks_list = [checkpoint, early]

loss_history = [model.fit_generator(train_datagen,
                                 epochs=MAX_TRAIN_EPOCHS,
                                 steps_per_epoch=STEPS_PER_EPOCH,
                                 validation_data=(valid_x, valid_y),
                                 callbacks=callbacks_list,
                                workers=1 # the generator is not very thread safe
                                           )]
model.load_weights(weight_path)
model.save('model.h5')
epochs = np.concatenate([mh.epoch for mh in loss_history])
loss = np.concatenate([mh.history['loss'] for mh in loss_history])
val_loss  = np.concatenate([mh.history['val_loss'] for mh in loss_history])
train_accuracy = np.concatenate([mh.history['top_3_accuracy'] for mh in loss_history])
test_accuracy = np.concatenate([mh.history['val_top_3_accuracy'] for mh in loss_history])
print ('train accuray: {}'.format(max(train_accuracy)))
print ('test accuray: {}'.format(max(test_accuracy)))
#### Model performance
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (30,10))

ax1.plot(epochs,train_accuracy, epochs,test_accuracy)
ax1.legend(['Training', 'Validation'])
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.set_title('accuracy train vs validation')

ax2.plot(epochs,loss, epochs,val_loss)
ax2.legend(['Training', 'Validation'])
ax2.set_xlabel('epoch')
ax2.set_ylabel('loss')
ax2.set_title('loss train vs validation')

valid_set = pd.read_csv(os.path.join(CSV_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=16)
valid_x = df_to_image_array_xd(valid_set, size)
valid_y = to_categorical(word_encoder.transform(valid_set["word"].values),num_classes=340).astype(np.int32)
valid_img_label = model.predict(valid_x, verbose=True)
top_3_pred_valid = [word_encoder.classes_[np.argsort(-1*c_pred)[:3]] for c_pred in valid_img_label]
top_3_pred_valid = [' '.join([col.replace(' ', '_') for col in row]) for row in top_3_pred_valid]
fig, m_axs = plt.subplots(4,4, figsize = (20, 20))
rand_idxs = np.random.choice(range(valid_x.shape[0]), size = 16, replace=False)
for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
    test_arr = valid_x[c_id, :, :, 0]
    c_ax.imshow(test_arr,cmap=plt.cm.gray)
    c_ax.axis('off')
    c_ax.set_title((top_3_pred_valid[c_id],valid_set["word"].iloc[c_id]))


