# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import pyplot as plt

%matplotlib inline

import  glob, cv2, os, random



from tensorflow import keras

from tensorflow.keras import layers



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
submission = pd.read_csv('/kaggle/input/recunoasterea-scris-de-mana/sampleSubmission.csv')#

submission.head()
train = pd.read_csv('/kaggle/input/recunoasterea-scris-de-mana/train.csv')

train.head()
train['label'].hist()
test_imgs = glob.glob('/kaggle/input/recunoasterea-scris-de-mana/test/test/*')

train_imgs = glob.glob('/kaggle/input/recunoasterea-scris-de-mana/train/train/*')

len(test_img), len(train_img)
img2text = {i:str(j) for i,j in train.to_numpy()}

#img2text
from keras import backend as K

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Input, Dense, Activation

from keras.layers import Reshape, Lambda, BatchNormalization

from keras.layers.merge import add, concatenate

from keras.models import Model

from keras.layers.recurrent import LSTM



K.set_learning_phase(0)



#inspirat de la https://github.com/qjadud1994/CRNN-Keras/



# # Loss and train functions, network architecture

def ctc_lambda_func(args):

    y_pred, labels, input_length, label_length = args

    # the 2 is critical here since the first couple outputs of the RNN

    # tend to be garbage:

    y_pred = y_pred[:, 2:, :]

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)



img_w, img_h = 128, 32

num_classes = 13

max_text_len = 9



def get_Model(training):

    input_shape = (img_w, img_h, 1)     # (128, 32, 1)



    # Make Networkw

    inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 32, 1)



    # Convolution layer (VGG)

    inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)  # (None, 128, 32, 64)

    inner = BatchNormalization()(inner)

    inner = Activation('relu')(inner)

    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

    

    inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 64, 16, 64)

    inner = BatchNormalization()(inner)

    inner = Activation('relu')(inner)

    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None,64, 32, 64)

    

    inner = Conv2D(64, (2, 2), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 32, 8, 64)

    inner = BatchNormalization()(inner)

    inner = Activation('relu')(inner)

    inner = MaxPooling2D(pool_size=(2, 2), name='max3')(inner)  # (None,64, 32, 64)

    

#     inner = Conv2D(64, (2, 2), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, 32, 8, 64)

#     inner = BatchNormalization()(inner)

#     inner = Activation('relu')(inner)

#     inner = MaxPooling2D(pool_size=(2, 2), name='max4')(inner)  # (None,64, 32, 64)



    inner = Conv2D(128, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, 16, 4, 512)

    inner = BatchNormalization()(inner)

    inner = Activation('relu')(inner)



    print(inner)

    # CNN to RNN

    inner = Reshape(target_shape=((16, 4*128)), name='reshape')(inner)  # (None, 32, 2048)

    inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)



    # RNN layer

    lstm_1 = LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)  # (None, 32, 512)

    lstm_1b = LSTM(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)

    reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (lstm_1b)



    lstm1_merged = add([lstm_1, reversed_lstm_1b])  # (None, 32, 512)

    lstm1_merged = BatchNormalization()(lstm1_merged)



    # transforms RNN output to character activations:

    inner = Dense(num_classes, kernel_initializer='he_normal',name='dense2')(lstm1_merged) #(None, 32, 63)

    y_pred = Activation('softmax', name='softmax')(inner)



    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32') # (None ,8)

    input_length = Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)

    label_length = Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)



    # Keras doesn't currently support loss funcs with extra parameters

    # so CTC loss is implemented in a lambda layer

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)



    if training:

        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

    else:

        return Model(inputs=[inputs], outputs=y_pred)

    

from keras import backend as K

K.clear_session()

model = get_Model(training=True)

model.summary()
letters = [i for i in '!~0123456789']

# # Input data generator

def labels_to_text(labels):    

    return ''.join(list(map(lambda x: letters[int(x)], labels)))



def text_to_labels(text):     

    text = str(text)

    while len(text)<8:

        text = '~'+text

    return list(map(lambda x: letters.index(x), text))





class TextImageGenerator:

    def __init__(self, img_dirpath, img_w, img_h,

                 batch_size, downsample_factor, max_text_len=8, isTesting=False,

                 validationNr=False, validationNR=1500):

        self.img_h = img_h

        self.img_w = img_w

        self.batch_size = batch_size

        self.max_text_len = max_text_len

        self.downsample_factor = downsample_factor

        self.img_dirpath = img_dirpath                  # image dir path

        self.img_dir = os.listdir(self.img_dirpath)     # images list

        if validationNr:

            self.img_dir = self.img_dir[:validationNR]

        else:

            self.img_dir = self.img_dir[validationNR:]

        self.n = len(self.img_dir)                     # number of images

        self.indexes = list(range(self.n))

        self.cur_index = 0

        self.imgs = np.zeros((self.n, self.img_h, self.img_w))

        self.texts = []



    def build_data(self):

        print(self.n, " Image Loading start...")

        for i, img_file in enumerate(self.img_dir):

            img = cv2.imread(self.img_dirpath + img_file, cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (self.img_w, self.img_h))

            img = img.astype(np.float32)

            img = (img / 255.0) * 2.0 - 1.0



            self.imgs[i, :, :] = img

            self.texts.append(img2text[img_file.split('/')[-1]])

        print(len(self.texts) == self.n)

        print(self.n, " Image Loading finish...")



    def next_sample(self):      ## index max 

        self.cur_index += 1

        if self.cur_index >= self.n:

            self.cur_index = 0

            random.shuffle(self.indexes)

        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]



    def next_batch(self):      

        while True:

            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])     # (bs, 128, 64, 1)

            Y_data = np.ones([self.batch_size, self.max_text_len])             # (bs, 9)

            input_length = np.ones((self.batch_size, 1)) * 14#(self.img_w // self.downsample_factor - 2)  # (bs, 1)

            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)



            for i in range(self.batch_size):

                img, text = self.next_sample()

                img = img.T

                img = np.expand_dims(img, -1)

                X_data[i] = img

                Y_data[i] = text_to_labels(text)

                label_length[i] = 8# len(text)



            # dict 형태로 복사

            inputs = {

                'the_input': X_data,  # (bs, 128, 64, 1)

                'the_labels': Y_data,  # (bs, 8)

                'input_length': input_length,  # (bs, 1) -> 모든 원소 value = 30

                'label_length': label_length  # (bs, 1) -> 모든 원소 value = 8

            }

            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1) -> 모든 원소 0

            yield (inputs, outputs)
# model = get_Model(training=True)





mg_w, img_h, batch_size, downsample_factor = 128, 32, 10, 4

train_file_path = '/kaggle/input/recunoasterea-scris-de-mana/train/train/'

tiger_train = TextImageGenerator(train_file_path, img_w, img_h, batch_size, downsample_factor)

tiger_train.build_data()
g = tiger_train.next_batch()

inputs, outputs = next(g)
inputs.keys()
inputs['input_length']
inputs['the_input'].shape
plt.imshow(inputs['the_input'][0,:,:,0])
outputs
from keras import backend as K

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint





model = get_Model(training=True)



val_batch_size = 10



train_file_path = '/kaggle/input/recunoasterea-scris-de-mana/train/train/'

tiger_train = TextImageGenerator(train_file_path, img_w, img_h, batch_size, downsample_factor, validationNr=True, validationNR=1500)

tiger_train.build_data()



tiger_val = TextImageGenerator(train_file_path, img_w, img_h, val_batch_size, downsample_factor, validationNr=False, validationNR=1500)

tiger_val.build_data()



ada = Adam()



early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)

checkpoint = ModelCheckpoint(filepath='LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1)

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)



# captures output of softmax so we can decode the output during visualization

model.fit_generator(generator=tiger_train.next_batch(),

                    steps_per_epoch=int(tiger_train.n / batch_size),

                    epochs=60,

#                     callbacks=[checkpoint],

                    validation_data=tiger_val.next_batch(),

                    validation_steps=int(tiger_val.n / val_batch_size))

model.save_weights('LSTM+BN4--26--0.011.hdf5')


from keras import backend as K

K.clear_session()

model = get_Model(training=False)

model.load_weights('LSTM+BN4--26--0.011.hdf5')
import itertools, os, time



def decode_label(out):

    # out : (1, 32, 42)

    out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32

    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value

    outstr = ''

    for i in out_best:

        if i < len(letters) and letters[i] not in ['~','!']:

            outstr += letters[i]

    return outstr

img2text = {}

for i,test_img in enumerate(test_imgs[:]):#train_imgs:

    if i%100==0:

        print(i,'of', len(test_imgs))

    img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)



    img_pred = img.astype(np.float32)

    img_pred = cv2.resize(img_pred, (128, 32))

    img_pred = (img_pred / 255.0) * 2.0 - 1.0

    img_pred = img_pred.T

    img_pred = np.expand_dims(img_pred, axis=-1)

    img_pred = np.expand_dims(img_pred, axis=0)



    net_out_value = model.predict(img_pred)



    pred_texts = decode_label(net_out_value)

    img2text[test_img.split('/')[-1]] = 0 if len(pred_texts)==0 else pred_texts#int(pred_texts)

    if False:

        print(pred_texts)

        plt.imshow(img)

        print(img2text)

        break
#save the prediction

submission['Expected'] = submission['Id'].apply(lambda x: img2text[x])

submission.to_csv('submission.csv', index=False)

submission.head()