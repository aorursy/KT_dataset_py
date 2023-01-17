# Read image names

import pandas as pd

df = pd.read_csv(r"../input/newTrainLabels.csv")

df.head()
# Convert categorical level to binary

import numpy as np

df['level'] = 1*(df['level'] > 0)

df.head(10)
# Resample and take only 2000 first rows

df = df.sample(n = 2000, random_state = 2019)

df.head()
# Check the size

df.shape
# Create level histogram

df['level'].hist(bins = [0,1, 2], rwidth = 0.5, align = 'left');
# Create image data generator

from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

    rescale=1./255, 

    validation_split = 0.25)



# Data generator for training

train_generator = datagen.flow_from_dataframe(

    dataframe = df, 

    directory = "../input/300_train/300_train",

    has_ext = False,

    x_col = "image", 

    y_col = "level", 

    class_mode = "binary", 

    target_size = (100, 100), 

    batch_size = 16,

    subset = 'training')



# Data generator for validation

valid_generator = datagen.flow_from_dataframe(

    dataframe = df, 

    directory = "../input/300_train/300_train",

    has_ext = False,

    x_col = "image", 

    y_col = "level", 

    class_mode = "binary", 

    target_size = (100, 100), 

    batch_size = 16,

    subset = 'validation')
# Create a basic Sequential model with several Conv2D layers



from keras import Sequential

from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense

from keras import optimizers



model = Sequential()

# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.

# this applies 32 convolution filters of size 3x3 each.

# See: https://keras.io/getting-started/sequential-model-guide/ -> VGG-like convnet

model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (100, 100, 3)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation = 'sigmoid'))



sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

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

N = 10 # Number of epochs



# Train and count time

model_name = strftime('Case2-%Y-%m-%d-%H%M%S.h5', localtime())

t1 = time()

h = model.fit_generator(generator = train_generator,

                    steps_per_epoch = STEP_SIZE_TRAIN,

                    validation_data = valid_generator,

                    validation_steps = STEP_SIZE_VALID,

                    epochs = N,

                    verbose = 2)

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
# Calculate the true and predicted values

y_true = valid_generator.classes

predict = model.predict_generator(valid_generator)

y_pred = predict > 0.5
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
# Check the histogram of the predicted values

plt.hist(predict, bins = np.arange(0.275, 0.280, 0.0001));

# They are in narrow range !!!
# Try different decision point

dp = 0.2765

cm = confusion_matrix(y_true, predict > dp)

print('Confusion matrix:')

print(cm)

print('')



cr = classification_report(y_true, predict > dp)

print('Classification report:')

print(cr)

print('')
from sklearn.metrics import accuracy_score

a = accuracy_score(y_true, predict > dp)

print('Accuracy with decision point {:.4f} ==> {:.4f}'.format(0.2792, a))
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