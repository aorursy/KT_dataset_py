# hide harmless Python warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
from datetime import datetime, timedelta
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Dense, Dropout, MaxPooling2D, Flatten, Activation, Input
from keras.layers.convolutional import Conv2D
from keras.models import Model, model_from_json
from keras import backend as K
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
%matplotlib inline

from kerasbestfit import kbf  # read the above info on how to install this custom module.

# prevent Tensorflow from displaying harmless warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# set numpy to be able to display wider output without truncating
np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)

# set the paths and filenames
train_file_csv = "../input/train.csv"
test_file_csv = "../input/test.csv"
x_train_file_npy = "x_train.npy"
y_train_file_npy = "y_train.npy"
x_test_file_npy = "x_test.npy"
x_train_cleaned_file_npy = "x_train_cleaned.npy"
y_train_cleaned_file_npy = "y_train_cleaned.npy"
model_path = ""

# define a logger function.  This will log to the screen and/or file.
# we'll also pass this function as a parameter to the kerasbestfit function later
log_file = "log.txt"
log_mode = 'both'
def log_msg(msg=''):
    fmt = "%H:%M:%S"
    s = f'{datetime.today().strftime(fmt)}: {msg}'
    if log_mode == 'file_only' or log_mode == 'both':
        with open(log_file, "a") as myfile:
            myfile.write(f'{s}\n')
    if log_mode == 'screen_only' or log_mode == 'both':
        print(s)
        
log_msg('Script started.')        
# convert the kaggle input data csv's to faster .npy. Keep the data types as uint8 so the files are small as possible
# convert only if the converted files don't already exist
if not os.path.isfile(x_train_file_npy): 
    # convert the train.csv
    # the train.csv file has the first columns for the label and the remaining columns as pixel data.  
    # we'll use this data for training the model.
    mnist_train_dataset = read_csv(train_file_csv, delimiter=',').values
    # extract the first column.  This will be the labels and we'll call it y_train
    y_train = mnist_train_dataset[:, 0]
    y_train = y_train.astype('uint8')
    # extract the remaining columns. This will be images we'll call it x_train
    x_train = mnist_train_dataset[0:, 1:]
    x_train = x_train.astype('uint8')
    # save it
    np.save(x_train_file_npy, x_train)
    np.save(y_train_file_npy, y_train)
    
if not os.path.isfile(x_test_file_npy): 
    # convert the test.csv.  This file contains images only. It doesn't contain labels.  This data will be used
    # later on when we test the model for creating the submission.csv we'll send to Kaggle for scoring.
    mnist_test_dataset = read_csv(test_file_csv, delimiter=',').values
    x_test = mnist_test_dataset
    x_test = x_test.astype('uint8')
    np.save(x_test_file_npy, x_test)
    
    # we now have x_train.npy and y_train.npy for training, and x_test.npy for testing.  
# now let's load the training data from those fast .npy files.  Note how fast it loads!
x_train = np.load(file=x_train_file_npy)
y_train = np.load(file=y_train_file_npy)
# we will load the x_test file later, just before we need to do a prediction using our model
# print out the array dimensions so we can see how much data we have
log_msg(f'x_train: {x_train.shape}.  These are the training images.')
log_msg(f'y_train: {y_train.shape}.  These are the training labels.')
plt.imshow(x_train[8].reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
plt.show()
# let's take a look at some of these bad images
plt.imshow(x_train[35396].reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
plt.show()
# remove incorrect images from our training data set
lst = [12817,  # 0's
       60, 191, 2284, 2316, 5275, 7389, 19633, 19979, 24891, 29296, 32565, 38191, 38544, 40339, 41739,  # 1's
       4677, 7527, 9162, 13471, 16598, 20891, 27364,  # 2's
       240, 11593, 11896, 17966, 25708, 28560, 33198, 34477, 36018, 41492,  # 3's
       1383, 6781, 22478, 23604, 26171, 26182, 26411, 18593, 34862, 36051, 36241, 36830, 37544,  # 4's
       456, 2867, 2872, 5695, 6697, 9195, 18319, 19364, 27034, 29253, 35620,  # 5's
       7610, 12388, 12560, 14659, 15219, 18283, 24122, 31649, 40214, 40358, 40653,  # 6's
       6295, 7396, 15284, 19880, 20089, 21423, 25233, 26366, 26932, 27422, 31741,  # 7's
       8566, 10920, 23489, 25069, 28003, 28851, 30352, 30362, 35396, 36984, 39990, 40675, 40868, 41229,  # 8's
       631, 4226, 9943, 14914, 15065, 17300, 18316, 19399, 20003, 20018, 23135, 23732, 29524, 33641, 40881, 41354  # 9's
       ]
x_cleaned = np.delete(x_train, lst, 0)
y_cleaned = np.delete(y_train, lst, 0)
np.save(x_train_cleaned_file_npy, x_cleaned)
np.save(y_train_cleaned_file_npy, y_cleaned)
# reload the clean data from those fast .npy files
x_train = np.load(file=x_train_cleaned_file_npy)
y_train = np.load(file=y_train_cleaned_file_npy)
# count the number of training and test images
log_msg(f'There are {x_train.shape[0]} training images.')
# calculate class balance
log_msg('Class balance:')
unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, counts)))

# plot a barchart showing class balance and mean count
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Class Balance')
ax.set_ylabel('Count')
ax.set_xlabel('Class')
ax.annotate("Mean", xy=(0.012, 0.86), xycoords="axes fraction")
plt.xticks(unique, unique)
ax.axhline(counts.mean(), color='gray', linewidth=1)
ax.bar(unique, counts, color='orange', align="center")
plt.show()
# let's take a look at the training data shape and format
log_msg(f'x_train shape is {x_train.shape} of data type {x_train.dtype}.  These are our training IMAGES.')
log_msg(f'y_train shape is {y_train.shape} of data type {y_train.dtype}.  These are our training LABELS.')
# let's examine one of the training images. We'll pick the eleventh image in the array
# this is a snifftest to ensure that our clean dataset actually has valid data
sample = 10  # change this value to a different image
plt.title('Sample: %d  Label: %d' % (sample, y_train[sample]))
plt.imshow(x_train[sample].reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
plt.show()
# now let's split up our training data into TRAINING and VALIDATION datasets.
# the random seed ensures thta we get the same identical data split every time we run this function.
# you can also run this without a random seed to let it randomly split the data.
# The stratify parameter tells it to make the class balance ratios the same in the training and validation datasets.
# this avoids the situation of a digit becoming over/underrepresented in either dataset.
# for example, if 5's were only 8% of the training data, we sure don't want the validation dataset to only 
# have 3% of fives.  Stratify will keep it at 8%.
random_seed = np.random.seed(2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed,
                                          stratify=y_train) 
log_msg('')
log_msg('Changing data type to float and normalizing values...')
# we'll need change our data type to float and normalize the image data
# set datatypes to float32 and normalize
x_train = x_train.astype('float32')
x_train /= 255
x_val = x_val.astype('float32')
x_val /= 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
log_msg(f'x_train shape is {x_train.shape} of data type {x_train.dtype}.  These are our TRAINING images.')
log_msg(f'x_val shape is {x_val.shape} of data type {x_val.dtype}.  These are our VALIDATION images.')

# and we'll also need to convert our y_train labels float32 to one-hot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_train = y_train.astype('float32')
y_val = np_utils.to_categorical(y_val, 10)
y_val = y_val.astype('float32')
log_msg(f'y_train shape is {y_train.shape} of data type {y_train.dtype}.  These are our TRAINING labels.')
log_msg(f'y_val shape is {y_val.shape} of data type {y_val.dtype}.  These are our VALIDATION labels.')
# set do_training to False if you want to skip training and go straight to prediction.  This is useful if you already have 
# model files saved from a previous training run.
do_training = True  
if do_training:
    log_msg('---- TRAINING BEGIN ----')        
    # this is for formatting numbers in the log
    format_metric_val = '{:1.10f}'  

    # set this metric to val_acc for accuracy, and val_loss for loss.  Running val_acc is fine for this competition.
    metric = 'val_acc'              
    if metric == 'val_acc':
        best_metric_val_so_far = 0
        snifftest_max_epoch = 0
        snifftest_metric_val = 0
    elif metric == 'val_loss':
        best_metric_val_so_far = 100.0
        snifftest_max_epoch = 0
        snifftest_metric_val = 100.0
        
    iter_best_fit = 0
    # init timed session.  This allows you to set a training time limit.  Handy for Keras 6 hour limit.
    max_duration_mins = 330  # this is 330 minutes (5.5 hrs).  Set it to 0 if you don't want a timed session.
    fmt = "%a %b %d %H:%M:%S"
    if max_duration_mins == 0:
        started_at = 0
        finish_by = 0
    else:
        started_at = datetime.today()
        finish_by = started_at + timedelta(minutes=max_duration_mins)
        log_msg(f'Started at {started_at.strftime(fmt)}, finish by {finish_by.strftime(fmt)}')

    # run the training x times and save the model weights that give the best metric
    # you could set it to x = 200 and let it run for hours if you want.
    # note: you can't pause training, then resume it later due to the nature of the Adam optimizer,
    # so you must train it uninterrupted.
    # training will end after the max_duration_mins has passed or x iterations has completed, whichever happens first.
    x = 3
    epochs=500
    patience=30
    batch_size=500
    for xtr in range(0, x):
        K.clear_session()  #clears tensorflow resources
        log_msg(f'Iteration {xtr} of {x - 1}')
        # now we'll define our model.  
        input = Input(shape=(28, 28, 1))
        x1 = Conv2D(32, (5, 5), padding='same', kernel_initializer='he_normal')(input)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x1)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)
        x1 = Dropout(0.5)(x1)
        x1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(32, (5, 5), padding='same', kernel_initializer='he_normal')(x1)
        x1 = Activation('relu')(x1)
        x1 = Dropout(0.5)(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)
        x1 = Dropout(0.5)(x1)
        x1 = Flatten()(x1)
        x1 = Dense(128, activation='relu', kernel_initializer='he_normal')(x1)
        x1 = Dropout(0.5)(x1)
        output = Dense(10, activation='softmax', kernel_initializer='he_normal')(x1)
        model = Model(inputs=[input], outputs=[output])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #call the kerasbestfit.find_best_fit function.  It will save the model weights with the best metric
        results, log = kbf.find_best_fit(model=model, metric=metric, xtrain=x_train, ytrain=y_train, xval=x_val,
                                         yval=y_val, validation_split=0, batch_size=batch_size, epochs=epochs, 
                                         patience=patience, snifftest_max_epoch=snifftest_max_epoch,
                                         snifftest_metric_val=snifftest_metric_val,
                                         show_progress=True, format_metric_val=format_metric_val,
                                         save_best=True, save_path=model_path,
                                         best_metric_val_so_far=best_metric_val_so_far,
                                         logmsg_callback=log_msg, finish_by=finish_by)
        del model
        # notify if we found a new best metric val
        is_best = False
        if metric == 'val_acc':
            is_best = results['best_metric_val'] > best_metric_val_so_far
        elif metric == 'val_loss':
            is_best = results['best_metric_val'] < best_metric_val_so_far
        if is_best:
            iter_best_fit = xtr
            best_metric_val_so_far = results['best_metric_val']
            sbest_metric_val_so_far = metric + '=' + format_metric_val.format(best_metric_val_so_far)
            sbest_epoch = results['best_epoch']
            best_log=results['history']
            best_epoch=results['best_epoch']
            s = f'NEW BEST SO FAR: {sbest_metric_val_so_far} on epoch {sbest_epoch}\n'
            log_msg(s)

        if results['expired']: #timer has expired
            break

    log_msg(f'The best result is {sbest_metric_val_so_far}')        
    log_msg('---- TRAINING END ----')        
# generate plot or accuracy and loss
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(20,10))
plt.xlim(0.0, best_epoch + 2.0)
plt.ylim(0.0, 1.0)
plt.plot(best_log['acc'])
plt.plot(best_log['val_acc'])
plt.plot(best_log['loss'])
plt.plot(best_log['val_loss'])
plt.axvline(results['best_epoch'], 0, 1, color='k', linestyle='--')
plt.title(f'Best Result: {sbest_metric_val_so_far}')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['acc', 'val_acc', 'loss', 'val_loss'], loc='center right')
plt.show()
# load saved model
with open('.//model.json', 'r') as f:
    modelx = model_from_json(f.read())
modelx.load_weights('.//model.hdf5')
Y_pred = modelx.predict(x_val)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(y_val, axis=1)

# this confusion matrix code from: https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
cm = confusion_matrix(Y_true, Y_pred_classes)
classes = range(10)
normalize = False
cmap = plt.cm.Greens
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title('Confusion Matrix')
plt.colorbar()
plt.rcParams.update({'font.size': 10})
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    # val = cm[i, j]
    # val = 100.0 * val/cm[i, :].sum()
    # print(cm[i, j], val)
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
del modelx
# load test data
x_test = np.load(file=x_test_file_npy)
x_test = x_test.astype('float32')
x_test /= 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_test = None  # we don't have this data because we don't have the test labels available

# load saved model
with open('.//model.json', 'r') as f:
    modelx = model_from_json(f.read())
modelx.load_weights('.//model.hdf5')

# # try the model with the built-in Keras MNIST dataset
# log_msg('')
# log_msg('Trying the model with the Keras built-in MNIST test dataset...')
# # (x_keras_train, y_keras_train), (x_keras_test, y_keras_test) = mnist.load_data()
# # we would prefer to load direct from the keras library, but this may trigger a download
# # of the dataset, which Kaggle blocks.  So we will load the data from these .npy's
# # that I have prepared and uploaded to Kaggle.
# x_keras_train = np.load(file=x_keras_train_file)
# y_keras_train = np.load(file=y_keras_train_file)
# x_keras_test = np.load(file=x_keras_test_file)
# y_keras_test = np.load(file=y_keras_test_file)
# # change data type and normalize
# x_keras_test = x_keras_test.astype('float32')
# y_keras_test = y_keras_test.astype('float32')
# x_keras_test /= 255
# x_keras_test = x_keras_test.reshape(x_keras_test.shape[0], 28, 28, 1)
# # y_keras_test = np_utils.to_categorical(y_keras_test, 10)
# test_labels = modelx.predict(x_keras_test)
# predicted_classes = np.argmax(test_labels, axis=1)
# correct_indices = np.nonzero(predicted_classes == y_keras_test)[0]
# incorrect_indices = np.nonzero(predicted_classes != y_keras_test)[0]
# accuracy = len(correct_indices) / (len(correct_indices) + len(incorrect_indices))
# log_msg(f'Prediction is {accuracy}')

# predict a single test image
log_msg('Predicting a single test image...')
img = x_test[0]    #change this from 0 to anything between 0-23999
plt.imshow(img.reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
plt.show()
img = img.reshape(1, 28, 28, 1)
prediction = modelx.predict([img])
predicted_classes = np.argmax(prediction[0], axis=0)
log_msg('Probabilities')
for x in range(0,10):
    s = f'Class {x}: {format_metric_val.format(prediction[0,x])}'
    log_msg(s)
log_msg(f'Predicted class: {predicted_classes}')

# now let's predict all the test images and save the results to submission.csv
log_msg('Predicting all test images...')
test_labels = modelx.predict(x_test)
predicted_classes = np.argmax(test_labels, axis=1)
log_msg('Prediction complete.')
submission = pd.DataFrame({
    'ImageID': range(1, 28001),
    'Label': predicted_classes
})
submission.to_csv('submission.csv', index=False)
log_msg('Saved predictions as submission.csv')

log_msg('Script ended.') 
