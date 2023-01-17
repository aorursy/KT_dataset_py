# module import
import glob
import os
from PIL import Image, ImageOps
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
# input train data
train_png = glob.glob('../input/train/*/*.png')
label = []
img = []
for fn in train_png:
    if fn[-3:] != 'png':
        continue
    label.append(fn.split('/')[-2])
    new_img = Image.open(fn)
    img.append(ImageOps.fit(new_img, (48, 48), Image.ANTIALIAS).convert('RGB'))
# show first data
img[0]
# data labeling
imgs = np.array([np.array(im) for im in img])
imgs = imgs.reshape(imgs.shape[0], 48, 48, 3) / 255
lb = LabelBinarizer().fit(label)
label = lb.transform(label) 
# split data
trainX, validX, trainY, validY = train_test_split(imgs, label, test_size=0.05, random_state=42)
# import keras module
from keras.layers import Dropout, Input, Dense, Activation,GlobalMaxPooling2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
# modeling
IM_input = Input((48, 48, 3))
IM = Conv2D(16, (3, 3))(IM_input)
IM = BatchNormalization(axis = 3)(IM)
IM = Activation('relu')(IM)
IM = Conv2D(16, (3, 3))(IM)
IM = BatchNormalization(axis = 3)(IM)
IM = Activation('relu')(IM)
IM = MaxPooling2D((2, 2), strides=(2, 2))(IM)
IM = Conv2D(32, (3, 3))(IM)
IM = BatchNormalization(axis = 3)(IM)
IM = Activation('relu')(IM)
IM = Conv2D(32, (3, 3))(IM)
IM = BatchNormalization(axis = 3)(IM)
IM = Activation('relu')(IM)
IM = GlobalMaxPooling2D()(IM)

IM = Dense(64, activation='relu')(IM)
IM = Dropout(0.5)(IM)
IM = Dense(32, activation='relu')(IM)
IM = Dropout(0.5)(IM)
IM = Dense(12, activation='softmax')(IM)
model = Model(inputs=IM_input, outputs=IM)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4), metrics=['acc'])
# model fitting train data
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import ModelCheckpoint

batch_size = 64
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
earlystop = EarlyStopping(patience=200)
modelsave = ModelCheckpoint(
    filepath='model.h5', save_best_only=True, verbose=1)
model.fit(
    trainX, trainY, batch_size=batch_size,
    epochs=200, 
    validation_data=(validX, validY),
    callbacks=[annealer, earlystop, modelsave]
)
# model fitting test data 
test_png = glob.glob('../input/test/*.png')
test_imgs = []
names = []
for fn in test_png:
    if fn[-3:] != 'png':
        continue
    names.append(fn.split('/')[-1])
    new_img = Image.open(fn)
    test_img = ImageOps.fit(new_img, (48, 48), Image.ANTIALIAS).convert('RGB')
    test_imgs.append(test_img)
model = load_model('model.h5')
# make test_X and test_Y
timgs = np.array([np.array(im) for im in test_imgs])
test_X = timgs.reshape(timgs.shape[0], 48, 48, 3) / 255
yhat = model.predict(test_X)
test_Y = lb.inverse_transform(yhat)
# make submission file
import pandas as pd
df = pd.DataFrame(data={'file': names, 'species': test_Y})
df_sort = df.sort_values(by=['file'])
df_sort.to_csv('results.csv', index=False)