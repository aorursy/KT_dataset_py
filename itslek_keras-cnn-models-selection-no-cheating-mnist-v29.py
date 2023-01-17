import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler, EarlyStopping

from keras import backend as K

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold

import matplotlib.pyplot as plt

import seaborn as sns

import gc

import datetime



# To stop potential randomness

seed = 42

np.random.seed(seed)



import os

print(os.listdir("../input"))
sample = pd.read_csv("../input/sample_submission.csv") #28 * 28 pixel

train =  pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)

num_classes = 10



Y_ = train['label'] # target data

X = train.drop(['label'],axis = 1) #train data



X = X.values.reshape(X.shape[0], img_rows, img_cols, 1).astype('float32')/255

Y = keras.utils.to_categorical(Y_, num_classes)



X.shape, Y.shape
def model_initiation(load_model=False, model_name=''):

    # Based on Model from https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist

    model = Sequential()

    model.add(Conv2D(32, kernel_size = 3, padding='same', activation='relu', input_shape = (28, 28, 1)))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 3, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))



    model.add(Conv2D(64, kernel_size = 3, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 3, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))



    model.add(Conv2D(128, kernel_size = 4, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dropout(0.25))

    model.add(Dense(10, activation='softmax'))

    

    if load_model:

        model.load_weights(model_name)



    # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return(model)
# Model

def cnn_fit(X_train,y_train, X_test,y_test, batch_size=128, epochs = 100, verbose = 2, cnn=0):

    model = model_initiation(load_model=False,)

    # CREATE MORE IMAGES VIA DATA AUGMENTATION - by randomly rotating, scaling, and shifting images.

    datagen = ImageDataGenerator(rotation_range=10,

                               zoom_range=0.1,

                               width_shift_range=0.1,

                               height_shift_range=0.1)



    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                              patience=4, 

                                              verbose=verbose, 

                                              factor=0.75, 

                                              min_lr=0.00001)

    

    #annealer = LearningRateScheduler(lambda x: 1e-2 * 0.75 ** x)

    

    # IMORTANT PART

    # we save only the best epoch on accuracy

    file_name_model = ("model_"+str(cnn)+".hdf5")

    checkpointer = ModelCheckpoint(monitor='val_acc',

                                   filepath=("./"+file_name_model),

                                   verbose=0, save_best_only=True, mode='max')

    

    #earlystopper = EarlyStopping(monitor='val_loss', min_delta=0,

    #                             patience=17, verbose=1, mode='auto')



    history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                                epochs = epochs, 

                                validation_data = (X_test,y_test),

                                callbacks=[learning_rate_reduction, checkpointer],

                                steps_per_epoch=(len(X_train)//batch_size),

                                verbose = verbose,)



    #score = model.evaluate(X_test,y_test, verbose=0)

    #print('Test loss:', score[0], ' accuracy:', score[1])

    return(history, model)
# split data into training set and testing set

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.15, random_state=42)
%%time

history, model = cnn_fit(X_train, y_train, X_val, y_val, batch_size=64, epochs = 75, verbose = 2)
# plot the accuracy and loss in each process: training and validation

def plot_(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']



    loss = history.history['loss']

    val_loss = history.history['val_loss']



    f, [ax1, ax2] = plt.subplots(1,2, figsize=(15, 5))

    ax1.plot(range(len(acc)), acc, label="acc")

    ax1.plot(range(len(acc)), val_acc, label="val_acc")

    ax1.set_title("Training Accuracy vs Validation Accuracy")

    ax1.legend()



    ax2.plot(range(len(loss)), loss, label="loss")

    ax2.plot(range(len(loss)), val_loss, label="val_loss")

    ax2.set_title("Training Loss vs Validation Loss")

    ax2.legend()
plot_(history)
score = model.evaluate(X_val,y_val, verbose=0)

print('loss:', round(score[0], 5))

print('accuracy:', round(score[1], 5))
print('Load best Model')

K.clear_session()

del model

gc.collect()

model = model_initiation(load_model=True, model_name='model_0.hdf5')

score = model.evaluate(X_val,y_val, verbose=0)

print('loss:', round(score[0], 5))

print('accuracy:', round(score[1], 5))
print(test.shape)

X_sub = test.values.reshape(test.shape[0], img_rows, img_cols, 1).astype('float32')/255

total_add_models = 0



# add or not add predict from model 0

if (round(score[1], 5) > 0.996):

    results = model.predict(X_sub)

    print('add predict from model 0')

    total_add_models+=1

else: 

    results = np.zeros((test.shape[0], 10))

    print('skip predict from model 0')



K.clear_session()

del model

del history

gc.collect()
%%time

cnn=0

selection_threshold_acc = 0.996

n_splits = 8



for batch_size in [64,128]:

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=batch_size)

    print('START with batch_size:', batch_size)

    for train_index, test_index in skf.split(X, Y_):

        cnn+=1

        time_start = datetime.datetime.now()

        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]

        y_train, y_test = np.array(Y_)[train_index], np.array(Y_)[test_index]

        # для StratifiedKFold пришлось использовать костыль в Y

        y_train = keras.utils.to_categorical(y_train, num_classes)

        y_test = keras.utils.to_categorical(y_test, num_classes)

        #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_val.shape, y_val.shape, )



        history, model = cnn_fit(X_train, y_train, X_test, y_test, 

                                 batch_size=batch_size, epochs=100, verbose=0, cnn=cnn)

        

        print(cnn, 'CNN accuracy', \

                '| Train last =', round(history.history['acc'][-1],5), \

                '| Test max =', round(max(history.history['val_acc']),5), \

                '| Took Time: ', (datetime.datetime.now() - time_start),)

        

        # Models selection

        if round(max(history.history['val_acc']), 5) >= selection_threshold_acc:

            K.clear_session()

            del model

            model = model_initiation(load_model=True, model_name='model_'+str(cnn)+'.hdf5')

            results = results + model.predict(X_sub)

            total_add_models+=1

            print('+ ADD', cnn, 'CNN with accuracy:', round(max(history.history['val_acc']), 5), ' Total add models:', total_add_models)

        else:

            print('- Skip', cnn, 'CNN not enough accuracy:', round(max(history.history['val_acc']),5))



        K.clear_session()

        del model

        del history

        #gc.collect()

    print('='*40)

print('Total add models:', total_add_models)
print(results.shape)



# Sum predictions

test_labels = np.argmax(results, axis=1)

print(test_labels[:10])



# submission dataframe

sub_df = pd.concat([pd.Series(range(1, 28001), name="ImageId"), pd.Series(test_labels, name="Label")], axis=1)



print('save submission')

sub_df.to_csv('MNIST_submission_v29.csv', index=False)
# All models are saved - the submission is reproducible

print(os.listdir("./"))