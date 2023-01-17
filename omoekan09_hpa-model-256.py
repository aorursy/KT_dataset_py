import tensorflow as tf
from keras import backend as K

from keras.optimizers import Adam, SGD, Adagrad, Adadelta

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, CSVLogger

from keras.models import Model, Sequential, load_model, model_from_json

from keras.layers import Flatten, Dense, Activation, Input, Dropout, Activation, BatchNormalization, Reshape

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer
import random

import os

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import cv2
data_path = '../input'

train_path = os.path.join(data_path, 'train')

test_path = os.path.join(data_path,'test')

labels_path = os.path.join(data_path, 'train.csv')
os.listdir(data_path)
def pick_label(x):

    # x list of strings

    t  = x

    if '27' in t:

        return '27'

    elif '15' in t:

        return '15'

    elif '10' in t:

        return '10'

    elif '9' in t:

        return '9'

    elif '8' in t:

        return '8'

    elif '0' in t:

        return '0'

    elif '25' in t:

        return '25'

    else:

        return t[0]

    

    





train_labels = pd.read_csv(labels_path,index_col=False)

labels_dict = dict(zip(train_labels.values[ :  ,0], train_labels.values[ : , 1]))

t = {k:pick_label(v.split()) for (k, v) in labels_dict.items()}

train_labels['t'] = t.values()

t_arr = np.array(list(t.items()))



train_ids, val_ids = train_test_split(train_labels, stratify=t_arr[ :, 1],

                                        test_size=0.1, random_state=48)
labels = [item.split() for item in train_labels['Target']]



mlb = MultiLabelBinarizer()

mlb.fit(labels)

classes = mlb.classes_

y_val = mlb.transform([item.split() for item in val_ids['Target']])
y_val
def model(sample_shape):

    



    model = Sequential()



    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=sample_shape, name='conv1'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(32, kernel_size=(3, 3), name='conv2'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(32, kernel_size=(3, 3), name='conv2b'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))



    model.add(Conv2D(64, kernel_size=(3, 3), name='conv3'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(64, kernel_size=(3, 3), name='conv4'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))

    

    

    model.add(Conv2D(128, kernel_size=(3, 3), name='conv5'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=(3, 3), name='conv6'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))

    

    model.add(Conv2D(256, kernel_size=(3, 3), name='conv7'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=(3, 3), name='conv8'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(4096, name='fc1'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    

    model.add(Dense(28))

    model.add(Activation('sigmoid'))

    return model



def get_rgb_img(image_folder,img_id):

    img = []

    img.append(plt.imread(os.path.join(image_folder,img_id+'_red.png')))

    img.append(plt.imread(os.path.join(image_folder, img_id+'_blue.png')))

    img.append(plt.imread(os.path.join(image_folder, img_id+'_green.png')))

    return np.stack(img, axis=2)
img = get_rgb_img(train_path,val_ids.values[0][0])

plt.imshow(img)

plt.show()
def val_generator(BATCH_SIZE):

   



    image_folder = train_path

    while True:

        

        

        val_imgs = []

        val_labels = []

        

        

        for f in val_ids.values:

            img = get_rgb_img(image_folder,f[0])

            val_imgs.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))

            val_labels.append(f[1])

            if len(val_imgs) == BATCH_SIZE:

                imgs = np.stack(val_imgs, axis=0)

                labels = mlb.transform([item.split() for item in val_labels])

                if len(imgs.shape[ 1: ]) == 2:

                    imgs = np.expand_dims(imgs, axis=3)

                yield (imgs, labels)

                val_imgs =[]

                val_labels =[]

        if len(val_imgs) > 0:

            imgs = np.stack(val_imgs, axis=0)

            labels = mlb.transform([item.split() for item in val_labels])

            if len(imgs.shape[ 1: ]) == 2:

                imgs = np.expand_dims(imgs, axis=3)

            yield (imgs, labels)

  
def train_generator(BATCH_SIZE):

    import random

    image_folder = train_path

    while True:

        #sample = train_ids.values

        #np.random.shuffle(sample)

        

        train_imgs = []

        train_labels = []

        

        for f in train_ids.values:

            img = get_rgb_img(image_folder,f[0])

            train_imgs.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))

            train_labels.append(f[1])

            if len(train_imgs) == BATCH_SIZE:

                imgs = np.stack(train_imgs, axis=0)

                labels = mlb.transform([item.split() for item in train_labels])

                if len(imgs.shape[ 1: ]) == 2:

                    imgs = np.expand_dims(imgs, axis=3)

                yield (imgs, labels)

                train_imgs = []

                train_labels = []

        if len(train_imgs) > 0:

            imgs = np.stack(train_imgs, axis=0)

            labels = mlb.transform([item.split() for item in train_labels])

            if len(imgs.shape[ 1: ]) == 2:

                imgs = np.expand_dims(imgs, axis=3)

            

            yield (imgs, labels)

  
## model parameters

DEPTH = 3

BATCH_SIZE = 32

IMG_SIZE = 256

SAMPLE_SHAPE = (IMG_SIZE, IMG_SIZE, DEPTH)



SEED = 1234

random.seed(SEED)
K.clear_session()

model = model(SAMPLE_SHAPE)
EPOCHS = 50

val_steps = int(np.ceil(len(val_ids)/BATCH_SIZE))

num_steps = int(np.ceil(len(train_ids)/BATCH_SIZE))

print('train_size: ', len(train_ids), 'batch: ', BATCH_SIZE, 'num steps: ', num_steps)
def f1(y_true, y_pred, thresholds= np.array(28*[0.5])):

    m, n = y_true.shape

    p = tf.cast(tf.greater(y_pred, thresholds), tf.float32)

    tp = tf.reduce_sum(y_true * p, 0)

    num_pos = tf.reduce_sum(tf.cast(y_true, tf.float32), 0)

    pred_pos = tf.reduce_sum(p, 0)

    precision = tp/(pred_pos + K.epsilon())

    recall = tp /(num_pos + K.epsilon())

    f1 = tf.reduce_mean(tf.divide(2*precision*recall, (precision + recall + K.epsilon())))

    K.get_session().run(tf.local_variables_initializer())

    return f1

lr = 1e-3

adam = Adam(lr=lr)

model.compile(optimizer=adam, 

                  loss='binary_crossentropy',

                metrics=[f1])





earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, mode='min', patience=6, verbose=0,restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min',factor=0.2, patience=3, min_lr = 1e-6, cooldown=1,verbose=1)



history = model.fit_generator(train_generator(BATCH_SIZE),

                                  steps_per_epoch = num_steps,

                                  validation_data=val_generator(BATCH_SIZE),

                                  validation_steps=val_steps,

                                  epochs=EPOCHS,

                                  callbacks=[earlyStopping, reduce_lr], verbose=1)
predictions = model.predict_generator(val_generator(BATCH_SIZE),steps=val_steps)

with tf.Session() as sess:

    print(sess.run(f1(y_val, predictions)))
submission = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'), index_col=False)
submission.head()
def test_generator(BATCH_SIZE):

    

    

    image_folder = test_path

    

    while True:

        

        

        test_imgs = []

        

        for f in submission['Id']:

            

            img = get_rgb_img(image_folder, f)

            test_imgs.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))

            

            if len(test_imgs) == BATCH_SIZE:

                imgs = np.stack(test_imgs, axis=0)

                if len(imgs.shape[ 1: ]) == 2:

                    imgs = np.expand_dims(imgs, axis=3)

                yield imgs

                test_imgs =[]

        if len(test_imgs) > 0:

            imgs = np.stack(test_imgs, axis=0)

            if len(imgs.shape[ 1: ]) == 2:

                imgs = np.expand_dims(imgs, axis=3)

            yield imgs

  
num_steps = int(np.ceil(len(submission)/BATCH_SIZE))

test_pred = model.predict_generator(test_generator(BATCH_SIZE), steps=num_steps )
submission.shape
def binary_prf1(y_true, y_pred):

    

    # y_true (ground truth)  - 1d array of 1, 0 

    # y_pred (predictions) - id arry of 1, 0

    # 1 - positive , 0 - negative

    num_pos = np.sum(y_true)

    pred_pos = np.sum(y_pred)

    tp = np.sum(y_true * y_pred)

    if pred_pos > 0:

        precision = tp/pred_pos

    else:

        precision = 0

    if num_pos > 0:

        recall = tp/num_pos

    else:

        recall = 0

        print('no pos cases for this class')

    if precision >0 or recall > 0:

        f1 = 2*precision*recall/(precision + recall)

    else:

        f1 = 0

    return precision, recall, f1





def max_thresh(y_val, predictions, n=100):

    x = np.linspace(0,1,n+1)[1 : -1]

    f1_matrix = np.zeros((len(x), 28))



    for i in range(28):

        class_f1 = []

        for thresh in x:

            pred_class = (predictions > thresh).astype(int)

            class_f1.append((binary_prf1(y_val[ :, i], pred_class[ : , i])[2]))

        f1_matrix[ :, i] = np.array(class_f1)

    #np.round(np.max(f1_matrix, axis=0),3)

    #f1_max = np.max(f1_matrix, axis=0)

    max_loc = np.argmax(f1_matrix, axis=0)

    max_thresh = [x[i] for i in max_loc]

    #print(max_thresh)

    #pc = (predictions > max_thresh).astype(int)

    return max_thresh
max_t = max_thresh(y_val, predictions)

with tf.Session() as sess:

    print(sess.run(f1(y_val, predictions, thresholds=max_t)))


pred_classes = (test_pred > max_t).astype(int)
pred_labels = mlb.inverse_transform(pred_classes)

pred_labels = [' '.join(item) for item in pred_labels]

submission['Predicted'] = pd.Series(pred_labels)
submission.head()
np.sum(submission["Predicted"] == '')
submission[submission["Predicted"] == ''] = '0 25'
submission.to_csv('submission.csv', index=False)