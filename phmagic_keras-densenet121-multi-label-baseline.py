import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import os
import pickle
from collections import defaultdict
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers import *
from sklearn.model_selection import train_test_split
from keras.callbacks import *

# from keras.utils import multi_gpu_model
DATA_DIR = '../input' #Or wherever your data is
# Test with training the stage 1 tuning labels (come up with your own labels)
CHALLENGE_DATA_DIR = DATA_DIR + '/'
IMG_DIR = CHALLENGE_DATA_DIR + '/stage_1_test_images/stage_1_test_images'
tuning_labels = pd.read_csv(CHALLENGE_DATA_DIR + '/tuning_labels.csv', 
                            names=['ImageID', 'Caption'],
                            index_col=['ImageID'])
tuning_labels.head()
tuning_labels_freq = defaultdict(int)

for r in tuning_labels['Caption']:
    labels = r.split()
    for l in labels:
        tuning_labels_freq[l] += 1

tuning_labels_list = list(tuning_labels_freq.keys())
print('Unique tuning labels', len(tuning_labels_list))

label_2_idx = {}
idx_2_label = {}
for i,v in enumerate(tuning_labels_list):
    label_2_idx[v] = i
    idx_2_label[i] = v
class_descriptions = pd.read_csv(CHALLENGE_DATA_DIR + '/class-descriptions.csv', index_col='label_code')

class_descriptions.loc['/m/0104x9kv']['description']
all_img_ids = list(tuning_labels.index.unique())
train_ids, test_ids = train_test_split(all_img_ids, test_size=0.01, random_state=21)
train_ids, valid_ids = train_test_split(train_ids, test_size=0.1, random_state=21)

print('Training on {} samples'.format(len(train_ids)))
print('Validating on {} samples'.format(len(valid_ids)))
print('Testing on {} samples'.format(len(test_ids)))
N_CLASSES = len(label_2_idx)
BATCH_SIZE = 8
INPUT_SIZE = 224
print(N_CLASSES)
def caption_2_one_hot(caption, n_classes=1, lookup_dict=None):
    y = np.zeros((n_classes))
    for w in caption.split():
        idx = lookup_dict[w]
        y[idx] = 1
    return y
def ImageDataGen(ids, df,
                 lookup_dict=label_2_idx,
                 n_classes=N_CLASSES,
                 img_dir=IMG_DIR, input_size=INPUT_SIZE,
                 bs=BATCH_SIZE, returnIds=False):
    while True:
        for start in range(0, len(ids), bs):
            x_batch = []
            y_batch = []
            end = min(start+bs, len(ids))
            sample = ids[start:end]
            for img_id in sample:
                img = cv.imread('{}/{}.jpg'.format(img_dir, img_id))
                if img is not None:
                    img = cv.resize(img, (input_size, input_size))
                    img = preprocess_input(img.astype(np.float32))
                    x_batch.append(img)
                    caption = df.loc[img_id]['Caption']
                    y = caption_2_one_hot(caption, n_classes=n_classes, lookup_dict=lookup_dict)
                    y_batch.append(y)
                    
            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            
            if returnIds:
                yield x_batch, y_batch, sample
            else:
                yield x_batch, y_batch
test_gen = ImageDataGen(test_ids, tuning_labels)
test_batch = next(test_gen)
fig = plt.figure(figsize=(20, 8))
for sample_idx in range(BATCH_SIZE):
    ax = fig.add_subplot(3,3, sample_idx + 1)
    ax.set_title(','.join([class_descriptions.loc[idx_2_label[i]]['description'] for i in (np.argwhere(test_batch[1][sample_idx]>0)).flatten()]))
    ax.imshow(test_batch[0][sample_idx])
    ax.set_axis_off()
plt.show()
    
def ClsModel(n_classes=1, input_shape=(224,224,3)):
    base_model = DenseNet121(weights=None, include_top=False, input_shape=input_shape)
    x = AveragePooling2D(pool_size=(3,3), name='avg_pool')(base_model.output)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='dense_post_pool')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_classes, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=base_model.input, output=output)
    return model
model = ClsModel(N_CLASSES)
model.summary()
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model_checkpoint = ModelCheckpoint(('./densenet.{epoch:02d}.hdf5'),
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=True)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                         patience=2, verbose=1)

callbacks = [model_checkpoint, reduce_learning_rate]
train_gen = ImageDataGen(train_ids, tuning_labels)
valid_gen = ImageDataGen(valid_ids, tuning_labels)
model.fit_generator(generator=train_gen, 
                    epochs=25, 
                    steps_per_epoch=np.ceil(len(train_ids)/BATCH_SIZE),
                   callbacks=callbacks,
                    validation_data=valid_gen,
                    validation_steps=np.ceil(len(valid_ids) / BATCH_SIZE))
test_preds = model.predict(test_batch[0])
fig = plt.figure(figsize=(20, 8))
pred_cutoff = 0.2
for sample_idx in range(BATCH_SIZE):
    ax = fig.add_subplot(3,3, sample_idx + 1)
    ax.set_title(','.join([class_descriptions.loc[idx_2_label[i]]['description'] for i in (np.argwhere(test_preds[sample_idx]>pred_cutoff)).flatten()]))
    ax.imshow(test_batch[0][sample_idx])
    ax.set_axis_off()
plt.show()
    
