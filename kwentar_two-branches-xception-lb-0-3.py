import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io

from scipy.misc import imread, imresize
from skimage.transform import resize
from tqdm import tqdm

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception
from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Dense, Multiply, Input
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam  
from keras import backend as K

from itertools import chain
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
path_to_train = '../input/train/'
data = pd.read_csv('../input/train.csv')

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)
class DataGenerator:
    def __init__(self):
        self.image_generator = ImageDataGenerator(rescale=1. / 255,
                                     vertical_flip=True,
                                     horizontal_flip=True,
                                     rotation_range=180,
                                     fill_mode='reflect')
    def create_train(self, dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 3
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images1 = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_images2 = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(random_indexes):
                image1, image2 = self.load_image(
                    dataset_info[idx]['path'], shape)
                batch_images1[i] = image1
                batch_images2[i] = image2
                batch_labels[i][dataset_info[idx]['labels']] = 1
            yield [batch_images1, batch_images2], batch_labels
            
    
    def load_image(self, path, shape):
        image_red_ch = skimage.io.imread(path+'_red.png')
        image_yellow_ch = skimage.io.imread(path+'_yellow.png')
        image_green_ch = skimage.io.imread(path+'_green.png')
        image_blue_ch = skimage.io.imread(path+'_blue.png')

        image1 = np.stack((
            image_red_ch, 
            image_yellow_ch, 
            image_blue_ch), -1)
        image2 = np.stack((
            image_green_ch, 
            image_green_ch, 
            image_green_ch), -1)
        image1 = resize(image1, (shape[0], shape[1], 3), mode='reflect')
        image2 = resize(image2, (shape[0], shape[1], 3), mode='reflect')
        return image1.astype(np.float), image2.astype(np.float)
# create train datagen
train_datagen = DataGenerator()

generator = train_datagen.create_train(
    train_dataset_info, 5, (299,299,3))
images, labels = next(generator)
images1, images2 = images
fig, ax = plt.subplots(2,5,figsize=(25,15))
for i in range(5):
    ax[0, i].imshow(images1[i])
for i in range(5):
    ax[1, i].imshow(images2[i])
print('min: {0}, max: {1}'.format(images1.min(), images1.max()))
# from https://www.kaggle.com/kmader/rgb-transfer-learning-with-inceptionv3-for-protein
data['target_list'] = data['Target'].map(lambda x: [int(a) for a in x.split(' ')])
all_labels = list(chain.from_iterable(data['target_list'].values))
c_val = Counter(all_labels)
n_keys = c_val.keys()
max_idx = max(n_keys)
data['target_vec'] = data['target_list'].map(lambda ck: [i in ck for i in range(max_idx+1)])
from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(data, 
                 test_size = 0.2, 
                  # hack to make stratification work                  
                 stratify = data['Target'].map(lambda x: x[:3] if '27' not in x else '0'), random_state=42)
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')
train_df.to_csv('train_part.csv')
valid_df.to_csv('valid_part.csv')



train_dataset_info = []
for name, labels in zip(train_df['Id'], train_df['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)
valid_dataset_info = []
for name, labels in zip(valid_df['Id'], valid_df['Target'].str.split(' ')):
    valid_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
valid_dataset_info = np.array(valid_dataset_info)
print(train_dataset_info.shape, valid_dataset_info.shape)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
train_sum_vec = np.sum(np.stack(train_df['target_vec'].values, 0), 0)
valid_sum_vec = np.sum(np.stack(valid_df['target_vec'].values, 0), 0)
ax1.bar(n_keys, [train_sum_vec[k] for k in n_keys])
ax1.set_title('Training Distribution')
ax2.bar(n_keys, [valid_sum_vec[k] for k in n_keys])
_ = ax2.set_title('Validation Distribution')
def create_model(input_shape, n_out):
    inp_image = Input(shape=input_shape)
    inp_mask = Input(shape=input_shape)
    pretrain_model_image = Xception(
        include_top=False, 
        weights='imagenet', 
        pooling='max')
    pretrain_model_image.name='xception_image'
    pretrain_model_mask = Xception(
        include_top=False, 
        weights='imagenet',    
        pooling='max')
    pretrain_model_mask.name='xception_mask'
    
    
    x = Multiply()([pretrain_model_image(inp_image), pretrain_model_mask(inp_mask)])
    out = Dense(n_out, activation='sigmoid')(x)
    model = Model(inputs=[inp_image, inp_mask], outputs=[out])

    return model
import tensorflow as tf
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
keras.backend.clear_session()

model = create_model(
    input_shape=(299,299,3), 
    n_out=28)

model.compile(
    loss='binary_crossentropy', 
    optimizer='adam',
    metrics=['acc', f1])

model.summary()
epochs = 4; batch_size = 12
checkpointer = ModelCheckpoint(
    '../working/Xception.model', 
    verbose=2, 
    save_best_only=False)


# create train and valid datagens
train_generator = train_datagen.create_train(
    train_dataset_info, batch_size, (299,299,3))
validation_generator = train_datagen.create_train(
    valid_dataset_info, batch_size, (299,299,3))
K.set_value(model.optimizer.lr, 0.0002)
# train model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_df)//batch_size,
    validation_data=validation_generator,
    validation_steps=len(valid_df)//batch_size//10,
    epochs=epochs, 
    verbose=1,
    callbacks=[checkpointer])
fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('loss')
ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax[1].set_title('acc')
ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
ax[0].legend()
_ = ax[1].legend()
submit = pd.read_csv('../input/sample_submission.csv')
%%time
predicted = []
from tqdm import tqdm_notebook
for name in tqdm(submit['Id']):
    path = os.path.join('../input/test/', name)
    image1, image2 = train_datagen.load_image(path, (299,299,3))
    score_predict = model.predict([image1[np.newaxis], image2[np.newaxis]])[0]
    label_predict = np.arange(28)[score_predict>=0.5]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)
submit['Predicted'] = predicted
submit.to_csv('submission.csv', index=False)
name_label_dict = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}
submit['target_list'] = submit['Predicted'].map(lambda x: [int(a) for a in str(x).split(' ')])
submit['target_vec'] = submit['target_list'].map(lambda ck: [i in ck for i in range(max_idx+1)])
all_labels = list(chain.from_iterable(submit['target_list'].values))
c_val = Counter(all_labels)
n_keys = c_val.keys()
max_idx = max(n_keys)
for k,v in name_label_dict.items():
    print(v, 'count:', c_val[k] if k in c_val else 0)
train_sum_vec = np.sum(np.stack(submit['target_vec'].values, 0), 0)
_ = plt.bar(n_keys, [train_sum_vec[k] for k in n_keys])
