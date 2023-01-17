import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
%matplotlib inline

root_path = Path('../TWaia')
train = root_path / 'train'
test = root_path / 'test'
from skimage import io
from skimage.transform import resize
from skimage.color import gray2rgb
from random import shuffle

label = []
img = []

for cate in os.listdir(train):
    for idx in os.listdir(train / cate):
        if idx != '.ipynb_checkpoints':
            label.append(cate)
            img.append(resize(gray2rgb(io.imread(train / cate / idx)), (224, 224, 3), mode='edge'))
from sklearn.preprocessing import LabelEncoder

mapping = pd.read_csv(root_path / 'mid_term_mapping.txt', header=None, names=['cate', 'code'])
mapping.sort_index(by='code', inplace=True)

le = LabelEncoder().fit(mapping.cate)
labels = le.transform(label)
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)

for train_idx, val_idx in skf.split(img, labels):
    break
    
print('training size:', len(train_idx))
print('validation size:', len(val_idx))
train_img = []
train_lab = []
val_img = []
val_lab = []

# train
shuffle(train_idx)
for i in train_idx:
    train_img.append(img[i])
    train_lab.append(labels[i])
    
# val
shuffle(val_idx)
for i in val_idx:
    val_img.append(img[i])
    val_lab.append(labels[i])
import keras.backend as K
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

K.clear_session()
basenet = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(basenet.output)
x = Dense(15, activation='softmax')(x)

model = Model(inputs=basenet.input, outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
from keras.utils import to_categorical

batch_size = 16

train_hist = model.fit(np.array(train_img), to_categorical(train_lab), 
                       batch_size=batch_size, 
                       epochs=2**4, 
                       validation_data=(np.array(val_img), to_categorical(val_lab)), 
                       verbose=2)
test_img = []

for idx in os.listdir(test):
    if idx != '.ipynb_checkpoints':
        test_img.append(resize(gray2rgb(io.imread(test / idx)), (224, 224, 3), mode='edge'))

# model prediction
pred = model.predict(np.array(test_img))
pred_cate = le.inverse_transform(pred.argmax(-1))
pred_final = pd.concat([pd.DataFrame(os.listdir(test)), pd.DataFrame(pred_cate)], 1)
pred_final.sample(10)
pred_final.to_csv(root_path / 'submit01.csv', index=False, header=None)
