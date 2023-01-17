# Read image names

import pandas as pd

df = pd.read_csv(r"../input/preprocessed-diabetic-retinopathy-trainset/newTrainLabels.csv")

df.head()
# Convert categorical level to binary and add file extensions

import numpy as np

df['level'] = 1*(df['level'] > 0)

df['image'] = df.image+'.jpeg'

print(df['image'])

print(df.head(10))
# check the datatype of level column

print(df.dtypes)

print(df['level'])
# Resample and take first rows

samples = 6000

df = df.sample(n = samples, random_state = 1897-77)

print(df.head())

df.reset_index(drop=True, inplace=True) # I've had troubles with shuffled dataframes on some earlier Cognitive Mathematics labs,

                                        # and this seemed to prevent those



# Check the size

df.shape

# Create samples histogram

df['level'].hist(bins = [0,1, 2], rwidth = 0.5, align = 'left')





# here we can split df into train/validation dataframes, and then we can use

# separated imagedatagenerators, such that we are able to use data augmentation 

# for only the traindata images

# the validdata images should be left raw as they are



setBoundary = round(samples * 0.67)

train_df = df[:setBoundary]

validate_df = df[setBoundary:]



train_df.reset_index(drop=True, inplace=True)

validate_df.reset_index(drop=True, inplace=True)







# train_df histogram

train_df['level'].hist(bins = [0,1, 2], rwidth = 0.5, align = 'left')





# validate_df histogram

validate_df['level'].hist(bins = [0,1, 2], rwidth = 0.5, align = 'left')

# Here the issue with original demo4 code from Sakari was that there was the typeError from imageDataGenerators

# at least for myself, so that the error message suggested that I should convert level column to str





df["level"]= df["level"].astype(str)

train_df["level"]= train_df["level"].astype(str)

validate_df["level"]= validate_df["level"].astype(str)



print(train_df.head())

print(validate_df.head())
# Create image data generator

from keras.preprocessing.image import ImageDataGenerator



# I think that it would be possible also to use data augmentation in the training_generator only.

# Also the interview of kaggle contestants showed my own suspicions to be true that one should not use 

# image shear with eye data (the perfectly healthy human eyeball should be round, so you dont get eyeglasses)

# those contestant used rotation and mirrorings of the image as I recall, but small amounts of

#  zoom would not be too bad either, I reckon



## use data augmentation for training

traingen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=15,

    zoom_range=0.05,

    horizontal_flip=True,

    vertical_flip=True,

    )



## just take the raw data for validation

validgen = ImageDataGenerator(

    rescale=1./255, 

    )



# Data flow for training

train_generator = traingen.flow_from_dataframe(

    dataframe = train_df, 

    directory =     "../input/preprocessed-diabetic-retinopathy-trainset/300_train/300_train",

    x_col = "image", 

    y_col = "level", 

    class_mode = "binary", 

    target_size = (100, 100), 

    batch_size = 32,

    shuffle = True,

    seed = 51

    )



# Data flow for validation

valid_generator = validgen.flow_from_dataframe(

    dataframe = validate_df, 

    directory = "../input/preprocessed-diabetic-retinopathy-trainset/300_train/300_train",

    x_col = "image", 

    y_col = "level", 

    class_mode = "binary", 

    target_size = (100, 100), 

    batch_size = 32,

    shuffle = False, # validgen doesnt need shuffle I think, according to teacher's readymade convnet example (?)

    )



# prepare test_df and test ImageDataGenerator for later testing purposes...

# its technically possible that we get some same imagelabels from the retinopathysolution.csv

# but its unlikely, so that the test_df should be mostly different imagelabels than validationdata or trainingdata

# test_generator could be used to test alreadt existing models for example so you get another comparison

# than simply the validation_generator



testgen = ImageDataGenerator(

    rescale=1./255,

    )







test_df = pd.read_csv("../input/preprocessed-diabetic-retinopathy-trainset/retinopathy_solution.csv")

test_df = test_df.sample(n = samples, random_state = 356)

test_df['level'] = 1*(test_df['level'] > 0)

test_df['image'] = test_df.image+'.jpeg'

test_df["level"]= test_df["level"].astype(str)

test_df.reset_index(drop=True, inplace=True)





test_generator = testgen.flow_from_dataframe(

    dataframe = test_df,

    directory = "../input/preprocessed-diabetic-retinopathy-trainset/300_test/300_test",

    x_col = "image",

    y_col = "level",

    class_mode = "binary",

        target_size = (100, 100), 

    batch_size = 32,

    shuffle = False, 

)

print(test_df['image'])

print(test_df.head(10))
# Create a basic Sequential model with several Conv2D layers

# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.

# model is based on that Chollet's cats/dogs classification model that was in the pdf links for case 2 

# the link was about something like how to create convnet model from small datasets

# link is as follows

# https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.2-using-convnets-with-small-datasets.ipynb



from keras import Sequential

from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense

from keras import optimizers



model = Sequential()



model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (100, 100, 3)))

model.add(MaxPooling2D((2, 2)))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dropout(0.5))



model.add(Dense(512, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



# initialize multiple optimizers but you can only choose one at a time of course!

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

addyboi = optimizers.Adam(lr=0.01, decay=1e-5)

rmsprop = optimizers.RMSprop(lr=1e-4)

basic_rms = optimizers.RMSprop()



model.compile(optimizer = sgd,

             loss='binary_crossentropy', 

              metrics = ["accuracy"])



model.summary()

from time import time, localtime, strftime

# Testing with localtime and strftime

print(localtime())

print(strftime('%Y-%m-%d-%H%M%S', localtime()))
# Calculate how many batches are needed to go through whole train and validation set

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size



N = 40 # Number of epochs



# Train and count time

model_name = strftime('Case2-%Y-%m-%d-%H%M%SlatesCatsDogsConvnet_returnableModel.h5', localtime())

t1 = time()

h = model.fit_generator(generator = train_generator,

                    steps_per_epoch = STEP_SIZE_TRAIN,

                    validation_data = valid_generator,

                    validation_steps = STEP_SIZE_VALID,

                    epochs = N,

                    verbose = 1)

t2 = time()

elapsed_time = (t2 - t1)



# Save the model

model.save(model_name)

print('')

print('Model saved to file:', model_name)

print('')



# Print the total elapsed time and average time per epoch in format (hh:mm:ss)

t_total = strftime('%H:%M:%S', localtime(t2 - t1))

t_per_e = strftime('%H:%M:%S', localtime((t2 - t1)/N))

print('Total elapsed time for {:d} epochs: {:s}'.format(N, t_total))

print('Average time per epoch:             {:s}'.format(t_per_e))

# download kaggle trained older model, based on catsAndDogs convNet

# I was testing an older model, in an earlier version of this notebook, but apparently that code didnt work out so well,

# so I removed those results.

# but, if you wanted you could test with that rdymodel.predict_generator() 



from keras.models import load_model #needed to load older trained models from memory/directory (so you dont need to train)

rdymodel = load_model("../input/returnable-catsanddogs-convnet-model/Case2-2019-03-02-112827latesCatsDogsConvnet_REAL_FINAL_VERSIONv3.h5")



rdy_predicts = rdymodel.predict_generator(test_generator, steps= np.ceil(test_generator.n / test_generator.batch_size)   )

print(rdy_predicts) ## just checking that it works at all...
# get the currently trained model, and plot the accuracies and loss for training and validation





%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np



epochs = np.arange(N) + 1.0



f, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,7))



def plotter(ax, epochs, h, variable):

    ax.plot(epochs, h.history[variable], label = variable)

    ax.plot(epochs, h.history['val_' + variable], label = 'val_'+variable)

    ax.set_xlabel('Epochs')

    ax.legend()



plotter(ax1, epochs, h, 'acc')

plotter(ax2, epochs, h, 'loss')

plt.show()



# get true values

y_true = valid_generator.classes

print(y_true)

print("\n")




# note about predict_generator, 

# sometimes it happened that, if you put argument steps = STEP_SIZE_VALID, then

# that throws error because of mismatched steps amount somewhere, I think that the

# np.ceil(validgen.n/validgen.batch_size) seems to fix it



predict = model.predict_generator(valid_generator, steps= np.ceil(valid_generator.n / valid_generator.batch_size)   )

print(predict)

print("\n")




y_pred = 1*(predict > 0.5)

print(y_pred)

print("\n")

# Calculate and print the metrics results

from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report



cm = confusion_matrix(y_true, y_pred)

print('Confusion matrix:')

print(cm)

print('')



cr = classification_report(y_true, y_pred)

print('Classification report:')

print(cr)

print('')
from sklearn.metrics import accuracy_score

a = accuracy_score(y_true, (y_pred))

print(a)

print('Accuracy with old decision point {:.4f} ==> {:.4f}'.format(0.5, a))
# Check the histogram of the predicted values

plt.hist(predict, bins = np.arange(0.2, 0.3, 0.001));

# Try different decision point

dp = 0.24

cm = confusion_matrix(y_true, (predict > dp))

print((predict>dp))

print('')



print('Confusion matrix:')

print(cm)

print('')



cr = classification_report(y_true, (predict > dp))

print('Classification report:')

print(cr)

print('')
from sklearn.metrics import accuracy_score

a = accuracy_score(y_true, (predict > dp))

print(a)

print('Accuracy with new decision point {:.4f} ==> {:.4f}'.format(dp, a))
# Calculate and plot ROC-curve

# See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html

from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_true, predict) 



plt.plot(fpr, tpr, color='darkorange', lw = 2)

plt.plot([0, 1], [0, 1], color='navy', lw = 2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic curve')

plt.show()