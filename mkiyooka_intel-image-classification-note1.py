!ls /kaggle/input/intel-image-classification
from IPython.display import SVG
from IPython.display import Image, display_png, display_jpeg

import itertools as itr
import os
from random import randint
from pathlib import Path
from pprint import pprint
import time

import cv2
from keras import optimizers, utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
for dirname_, _, filenames in os.walk('/kaggle/input'):
    dirname = Path(dirname_)
    for i, filename in enumerate(filenames):
        if i >= 1:
            break
        print(dirname / filename)
from tqdm.notebook import tqdm
display_jpeg(Image('/kaggle/input/intel-image-classification/seg_train/seg_train/glacier/15074.jpg'))
class_s2i = {
    'buildings': 0,
    'forest': 1,
    'glacier': 2,
    'mountain': 3,
    'sea': 4,
    'street': 5}
class_i2s = {v:k for k,v in class_s2i.items()}

pprint(class_s2i)
pprint(class_i2s)
root_dir = Path('/kaggle/input/intel-image-classification')
train_dir = root_dir / 'seg_train/seg_train'
test_dir = root_dir / 'seg_test/seg_test'
def load_image(image_path):
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (150,150))
    return image

# load_image('/kaggle/input/intel-image-classification/seg_train/seg_train/glacier/10011.jpg')
def load_image_pathes(root_dir, filenum=False):
    filenames = []
    labels = []
    for key in class_s2i.keys():
        p = root_dir / key
        if filenum:
            filenames_tmp = [f for f in p.iterdir()][:filenum]
        else:
            filenames_tmp = [f for f in p.iterdir()]
        filenames += filenames_tmp
        labels += [class_s2i.get(key,0)] * len(filenames_tmp)
    return filenames, labels
def load_images(root_dir, filenum=False):
    filenames, labels = load_image_pathes(root_dir, filenum)
    images = []
    for p in tqdm(filenames):
        images.append(load_image(p))
    return np.array(images), np.array(labels)
def show_images(images, labels, nrow=3, ncol=3):
    fig, axes = plt.subplots(nrow, ncol)
    fig.subplots_adjust(0,0,3,3)
    for j, (iy, ix) in enumerate(itr.product(range(nrow), range(ncol))):
        if nrow == ncol == 1:
            ax = axes
        elif nrow == 1 or ncol == 1:
            ax = axes[iy+ix]
        else:
            ax = axes[iy,ix]
        r = images[j]
        ax.imshow(images[j])
        ax.set_title(class_i2s.get(labels[j],-1))
        ax.axis('off')
def make_model(dummy=True):
    if dummy:
        model = Sequential()

        model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))
        model.add(Conv2D(16,kernel_size=(3,3),activation='relu'))
        model.add(MaxPooling2D(5,5))
        model.add(Conv2D(16,kernel_size=(3,3),activation='relu'))
        model.add(Conv2D(16,kernel_size=(3,3),activation='relu'))
        model.add(Conv2D(16,kernel_size=(3,3),activation='relu'))
        model.add(Conv2D(16,kernel_size=(3,3),activation='relu'))
        model.add(MaxPooling2D(5,5))
        model.add(Flatten())
        model.add(Dense(16,activation='relu'))
        model.add(Dense(16,activation='relu'))
        model.add(Dense(32,activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(6,activation='softmax'))

        model.compile(optimizer=optimizers.Adam(lr=0.0001),
                      loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    else:
        model = Sequential()

        model.add(Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))
        model.add(Conv2D(180,kernel_size=(3,3),activation='relu'))
        model.add(MaxPooling2D(5,5))
        model.add(Conv2D(180,kernel_size=(3,3),activation='relu'))
        model.add(Conv2D(140,kernel_size=(3,3),activation='relu'))
        model.add(Conv2D(100,kernel_size=(3,3),activation='relu'))
        model.add(Conv2D(50,kernel_size=(3,3),activation='relu'))
        model.add(MaxPooling2D(5,5))
        model.add(Flatten())
        model.add(Dense(180,activation='relu'))
        model.add(Dense(100,activation='relu'))
        model.add(Dense(50,activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(6,activation='softmax'))

        model.compile(optimizer=optimizers.Adam(lr=0.0001),
                      loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model
def show_score(trained):
    h = trained.history
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4.8, 1.6), dpi=100)
    fig.subplots_adjust(1,1,3,3)
    n = len(h['accuracy'])
    ax = axes[0]
    ax.plot(h['accuracy'])
    ax.plot(h['val_accuracy'])
    ax.set_title('Model accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.grid()
    ax.legend(['Train', 'Test'], loc='upper left')

    ax = axes[1]
    ax.plot(h['loss'])
    ax.plot(h['val_loss'])
    ax.set_title('Model loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.set_xlim(0, n)
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.legend(['Train', 'Test'], loc='upper left') 
train_images, train_labels = load_images(train_dir, filenum=150)
train_images, train_labels = shuffle(train_images, train_labels, random_state=0)
print('Shape of images:', train_images.shape)
print('Shape of labels:', train_labels.shape)
show_images(images, labels, nrow=3, ncol=4)
model = make_model(dummy=True)

# model.summary()
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
# utils.plot_model(model,to_file='model.png',show_shapes=True)

trained = model.fit(images, labels, epochs=20, validation_split=0.30)
model = make_model(dummy=False)

# model.summary()
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
# utils.plot_model(model,to_file='model.png',show_shapes=True)

trained = model.fit(images, labels, epochs=35, validation_split=0.0)
show_score(trained)
test_images, test_labels = load_images(test_dir)
scores_ = model.evaluate(test_images,test_labels, verbose=1)
scores = {k:v for k,v in zip(model.metrics_names, scores_)}
scores
nrow, ncol = 3, 3
fig = plt.figure(figsize=(30, 30))
outer = gridspec.GridSpec(nrow, ncol, wspace=0.2, hspace=0.2)

for i in range(nrow * ncol):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1)
    r = randint(0,len(test_images))
    image = np.array([test_images[r]])
    label = test_labels[r]
    pred_label = model.predict_classes(image)[0]
    pred_class = class_i2s.get(pred_label)
    pred_prob = model.predict(image).reshape(6)
    for j in range(2):
        if (j%2) == 0:
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(image[0])
            ax.set_title(f'ans: {class_i2s[label]}\npred: {pred_class}')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
        else:
            ax = plt.Subplot(fig, inner[j])
            ax.bar([0,1,2,3,4,5], pred_prob)
            ans_prob = [0] * 6
            ans_prob[label-1] = 1
            ax.bar([0,1,2,3,4,5], pred_prob * ans_prob)
            ax.set_xticks([0,1,2,3,4,5])
            ax.set_xticklabels(['buildings',
                                'forest',
                                'glacier',
                                'mountain',
                                'sea',
                                'street'], rotation=45)
            fig.add_subplot(ax)

fig.show()
# import pickle
# p = Path('/kaggle/working/trained.pickle')

# if p.exists():
#     with open(p, 'rb') as f:
#         pickle.load(trained, f)
# else:
#     trained = model.fit(images,labels,epochs=35,validation_split=0.30)
#     with open(p, 'wb') as f:
#         pickle.dump(trained, f)
