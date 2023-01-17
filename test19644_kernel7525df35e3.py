import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns # data visualization



from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical



import tensorflow as tf

import keras

from keras.models import Sequential

from keras.layers import Conv2D, Dropout, Dense, Flatten, BatchNormalization, MaxPooling2D,LeakyReLU

from keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.utils.np_utils import to_categorical

from tensorflow.keras.regularizers import l2



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
model_name = 'k-mnist_trained_model.h5'

nets = 5

model = [0] *(nets)
raw_train = pd.read_csv('../input/Kannada-MNIST/train.csv')

raw_test = pd.read_csv('../input/Kannada-MNIST/test.csv')



import random

random.seed(0)

np.random.seed(0)

import tensorflow as tf

tf.compat.v1.set_random_seed(0)
def SelfKFold(raw_train):

    '''

    return n train and val dataset.

    '''

    from sklearn.model_selection import KFold

    raw_train_val = raw_train

    random_train_val = raw_train_val.sample(frac=1).reset_index(drop=True)#need fix np.random

    kfold = KFold(n_splits=5,random_state=None)

    train_val_data = raw_train_val.drop('label',axis = 1).values.astype(np.uint8)

    train_val_label = raw_train_val['label'].values.astype(np.uint8)

    print('data shape:', train_val_data.shape)

    print('label shape:', train_val_label.shape)

    train_total = []

    train_label_total = []

    val_total = []

    val_label_total = []

    for _,val_index in kfold.split(train_val_data,train_val_label):

        print('val data index:',val_index[0],val_index[-1])

        train_sub_0 = train_val_data[val_index[-1]+1:]

        train_sub_1 = train_val_data[:val_index[0]]

        val_sub = train_val_data[val_index[0]:val_index[-1]+1]

        train_total.append(np.concatenate((train_sub_0,train_sub_1),axis=0))

        val_total.append(val_sub)



        train_label_sub_0 = train_val_label[val_index[-1]+1:]

        train_label_sub_1 = train_val_label[:val_index[0]]

        val_label_sub = train_val_label[val_index[0]:val_index[-1]+1]

        train_label_total.append(np.concatenate((train_label_sub_0,train_label_sub_1),axis=0))

        val_label_total.append(val_label_sub)

    print('number of dataset:',len(val_total))

    return train_total,train_label_total,val_total,val_label_total
#x_train_, y_train_,x_val_, y_val_ =  SelfKFold(raw_train)

x_train_ = raw_train.drop('label',axis = 1).values

#x_train_test = raw_test.drop('id',axis = 1).values

#x_train_ = np.concatenate([x_train_,x_train_test],axis=0)

y_train_ = raw_train['label'].values

#y_train_ = np.concatenate([y_train_,results5000],axis=0)

print(x_train_.shape)

print(y_train_.shape)





for j in range(nets):

    model[j] = Sequential([

        Conv2D(64, (3,3), padding='same', input_shape=(28, 28, 1)),

        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

        LeakyReLU(alpha=0.1),

        Conv2D(64,  (3,3), padding='same'),

        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

        LeakyReLU(alpha=0.1),

        Conv2D(64,  (3,3), padding='same'),

        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

        LeakyReLU(alpha=0.1),



        MaxPooling2D(2, 2),

        Dropout(0.25),



        Conv2D(128, (3,3), padding='same'),

        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

        LeakyReLU(alpha=0.1),

        Conv2D(128, (3,3), padding='same'),

        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

        LeakyReLU(alpha=0.1),

        Conv2D(128, (3,3), padding='same'),

        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

        LeakyReLU(alpha=0.1),



        MaxPooling2D(2,2),

        Dropout(0.25),    



        Conv2D(256, (3,3), padding='same'),

        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

        LeakyReLU(alpha=0.1),

        Conv2D(256, (3,3), padding='same'),

        BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),##

        LeakyReLU(alpha=0.1),



        MaxPooling2D(2,2),

        Dropout(0.25),





        Flatten(),

        Dense(256),

        LeakyReLU(alpha=0.1),



        BatchNormalization(),

        Dense(10, activation='softmax')

    ])

    model[j].summary()

#     optimizer = RMSprop(learning_rate=0.0025,###########

#     rho=0.9,

#     momentum=0.1,

#     epsilon=1e-07,

#     centered=True,

#     name='RMSprop')

    optimizer = RMSprop(lr=0.0025)

    model[j].compile(loss= keras.losses.CategoricalCrossentropy(label_smoothing=0.1),

              optimizer=optimizer,

              metrics=['accuracy'])

    

    
batch_size = 1024

num_classes = 10

epochs = 56

datagen_train = ImageDataGenerator(rotation_range = 10,

                                   width_shift_range = 0.25,

                                   height_shift_range = 0.25,

                                   shear_range = 10,

                                   zoom_range = 0.40,

                                   horizontal_flip = False)



datagen_val = ImageDataGenerator() 

def schedule(epoch):

    if epoch >= 52:

        lr = 1e-5

    elif epoch >= 48:

        lr = 0.0025 * 0.25 * 0.25 * 0.25

    elif epoch >= 43:

        lr = 0.0025 * 0.25 * 0.25

    elif epoch >= 35:

        lr = 0.0025 * 0.25

    else:

        lr = 0.0025

    return lr



learning_rate_reduction = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)



# learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau( 

#     monitor='loss',    # Quantity to be monitored.

#     factor=0.25,       # Factor by which the learning rate will be reduced. new_lr = lr * factor

#     patience=2,        # The number of epochs with no improvement after which learning rate will be reduced.

#     verbose=1,         # 0: quiet - 1: update messages.

#     mode="auto",       # {auto, min, max}. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; 

#                        # in the max mode it will be reduced when the quantity monitored has stopped increasing; 

#                        # in auto mode, the direction is automatically inferred from the name of the monitored quantity.

#     min_delta=0.0001,  # threshold for measuring the new optimum, to only focus on significant changes.

#     cooldown=0,        # number of epochs to wait before resuming normal operation after learning rate (lr) has been reduced.

#     min_lr=0.00001     # lower bound on the learning rate.

#     )



#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300, restore_best_weights=True)

# for j in range(nets):

#     loadmodelname = "../input/kernel41bf03341f/weights_N" + str(j)

#     model[j].load_weights(loadmodelname)

#     savemodelname = "weights_N" + str(j)

#     model[j].save_weights(savemodelname)

#     print("Saving", loadmodelname, "back to", savemodelname)
from keras.callbacks import ModelCheckpoint

nets2train = 5

history = [0] * nets2train

for j in range(nets2train):

    x_train = x_train_.reshape(-1, 28, 28,1).astype('float32') / 255

   # print(x_train.shape)

    #x_val = x_val_[j].reshape(-1, 28, 28,1).astype('float32') / 255

    x_val = x_train[:1024,:,:,:]

    y_train = to_categorical(y_train_)

    #y_val = to_categorical(y_val_[j])

    y_val = y_train[:1024,:]

    #print(y_train.shape)

    modelfilename = "weights_N_" + str(j)

    #modelfilename_val = "weights_N_val_" + str(j)

    checkpoint = ModelCheckpoint(modelfilename, monitor='acc', verbose=0, save_best_only=False, save_weights_only=False, period=epochs)

    #checkpoint_val = ModelCheckpoint(modelfilename_val, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, period=epochs)

    history[j] = model[j].fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size),

                              steps_per_epoch=len(x_train)//batch_size,

                              epochs=epochs,

                              validation_data=(x_val, y_val),

                              validation_steps=1,

                              callbacks=[learning_rate_reduction,checkpoint],

                              verbose=2)

    #model[j].save_weights(modelfilename)

    print("CNN", j,": Training done.")
test5000 = pd.read_csv("../input/Kannada-MNIST/test.csv")

X_test5000 = test5000.drop(labels = ["id"],axis = 1)

X_test5000 = X_test5000 / 255.0

X_test5000 = X_test5000.values.reshape(-1,28,28,1)

TTA = 6

nets4predict = 5

datagen_test = [0]*TTA

results5000 = np.zeros( (X_test5000.shape[0], 10) ) 

allthree = 0

preds_tta = []

for each_test in range(TTA):

    if allthree :

        if each_test == 0:

            datagen_test[each_test] = ImageDataGenerator(#rotation_range = 10,

                                           #width_shift_range = 0.25,

                                           #height_shift_range = 0.25,

                                           #shear_range = 10,

                                           zoom_range = 0.4,

                                           horizontal_flip = False,

               

                                           )

        elif each_test == 1:

            datagen_test[each_test] = ImageDataGenerator(#rotation_range = 10,

                                           width_shift_range = 0.25,

                                           #height_shift_range = 0.25,

                                           #shear_range = 10,

                                           #zoom_range = 0.4,

                                           horizontal_flip = False,

             

                                           )

        elif each_test == 2:

            datagen_test[each_test] = ImageDataGenerator(#rotation_range = 10,

                                           #width_shift_range = 0.25,

                                           height_shift_range = 0.25,

                                           #shear_range = 10,

                                           #zoom_range = 0.4,

                                           horizontal_flip = False,

               

                                           )

        elif each_test == 3:

            datagen_test[each_test] = ImageDataGenerator(rotation_range = 10,

                                           #width_shift_range = 0.25,

                                           #height_shift_range = 0.25,

                                           #shear_range = 10,

                                           #zoom_range = 0.4,

                                           horizontal_flip = False,

                                                       

                                           )

        elif each_test == 4:

            datagen_test[each_test] = ImageDataGenerator(#rotation_range = 10,

                                           #width_shift_range = 0.25,

                                           #height_shift_range = 0.25,

                                           #shear_range = 10,

                                           zoom_range = 0.2,

                                           horizontal_flip = False,

               

                                           )

        elif each_test == 5:

            datagen_test[each_test] = ImageDataGenerator(#rotation_range = 10,

                                           #width_shift_range = 0.25,

                                           #height_shift_range = 0.25,

                                           #shear_range = 10,

                                           #zoom_range = 0.4,

                                           horizontal_flip = False,

                                           

                                           )

        test_generator = datagen_test[each_test].flow(

                X_test5000,

                shuffle = False,

                #class_mode='categorical',

                batch_size=500)

        

        for j in range(nets4predict):

            print("CNN",j)

            loadmodelname = "weights_N" + str(j)

            model[j].load_weights(loadmodelname)

            test_generator.reset()

            preds = model[j].predict_generator(

                generator=test_generator,

                steps =int(X_test5000.shape[0]/500),

            )

            print(preds.shape)

            preds_tta.append(preds)

            





#             for x_batch in datagen_test[each_test].flow(X_test5000,batch_size=5000,shuffle=False):

#                 print('here')

#                 ttt = x_batch

#                 break

#             results5000 = results5000 + model[j].predict(ttt)

results5000 = np.mean(preds_tta,axis = 0)

results5000 = np.argmax(results5000,axis = 1)



submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

submission['label'] = results5000

submission.to_csv("submission.csv",index=False)



print("DONE.")