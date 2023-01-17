%%script false --no-raise-error

# check if all works

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

with tf.device('/gpu:0'):

    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')

    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')

    c = tf.matmul(a, b)



with tf.Session() as sess:

    print (sess.run(c))
%%script false --no-raise-error

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import time



from sklearn.model_selection import train_test_split, StratifiedKFold



from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical

from keras.models import Sequential, load_model

from keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense

from keras.callbacks import LearningRateScheduler



import tensorflow as tf

# some settings to run model

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:

    try:

        # Restrict TensorFlow to only use the fourth GPU

        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')



        # Currently, memory growth needs to be the same across GPUs

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')

        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:

        # Memory growth must be set before GPUs have been initialized

        print(e)
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer//test.csv')
y = train.pop('label')





train_y = to_categorical(y)



train_X = train/255

test_X = test/255



train_X = train_X.to_numpy()

test_X = test_X.to_numpy()
# reshape for NN

train_X = train_X.reshape(-1,28,28,1)

test_X = test_X.reshape(-1,28,28,1)
train_X.shape, train_y.shape
sub_test_X, sub_train_X, sub_test_y, sub_train_y = train_test_split(train_X, train_y, 

                                                                    train_size=0.2, stratify=train_y)
sub_test_X.shape, sub_train_X.shape, sub_test_y.shape, sub_train_y.shape
img_gen = ImageDataGenerator(rotation_range = 12, width_shift_range=.12, height_shift_range=.12, 

                             zoom_range=.12) 
plt.imshow(img_gen.flow(sub_train_X[0].reshape(-1,28,28,1), sub_train_y[0].reshape(1,10)).next()[0].reshape((28,28)))
def build_model(save = False):

    model = Sequential()

    model.add(Conv2D(32, kernel_size = 3,  activation = 'relu',  input_shape = (28,28,1)))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 3,  activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 3,  activation = 'relu', padding = 'same'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(64, kernel_size = 3,  activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 3,  activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 3,  activation = 'relu', padding = 'same'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(128, kernel_size = 3,  activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(10,  activation ='softmax'))



    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def train_model(model, train_X, train_y, save = False):

    sub_test_X, sub_train_X, sub_test_y, sub_train_y = train_test_split(train_X, train_y, 

                                                                    train_size=0.2, stratify=train_y)

    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

    start = time.clock()

    model.fit_generator(img_gen.flow(sub_train_X, sub_train_y), epochs=10, 

                                 steps_per_epoch=sub_train_X.shape[0]//64, 

                                validation_data = (sub_test_X,sub_test_y), callbacks=[annealer])

    print(f'Overall it takes {time.clock()-start} sec')

    if save ==True:

        model.save('/kaggle/input/trained-model/model.h5')

        



def load_my_model():

    re_model = load_model('/kaggle/input/trained-model/first_model.h5')

    return re_model

num_models = 3

models = [0]*num_models

for i in range(num_models):

    models[i] = build_model()

    train_model(models[i],train_X, train_y)



prediction = np.zeros((test_X.shape[0],10))

for i in range(len(models)):

    prediction += models[i].predict(test_X)
predict = np.argmax(prediction, axis =1)

predict = np.vstack((np.arange(predict.shape[0])+1, predict)).T
submission = pd.DataFrame(data=predict, columns=['imageid', 'label'])
submission
submission.to_csv('submit.csv',index=False )
print('FINISHED')