# Import libraries, more to follow later ...

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import cv2

import time

import os
SIZE = 6000 # Total number of images pooled from 300_train

SPLIT = 4000 # Number of images used for training

EPOCHS = 200

IMAGE_SIZE = 600 
# Read the train image names and labels

df = pd.read_csv('../input/newTrainLabels.csv')



# Take SIZE number of samples from dataframe

df = df.sample(n = SIZE)



# Source directory

source_dir = '../input/300_train/300_train/'



# Create destination directory

dest_dir = './temp/'

try:

    os.mkdir(dest_dir)

    print('Created a directory:', dest_dir)

except:

    print(dest_dir, 'already exists!')



# Start timing

start = time.time()



# Crop and resize all images. Store them to dest_dir

print('Cropping and rescaling the images:')

for i, file in enumerate(df['image']):

    try:

        fname = source_dir + file + '.jpeg'

        img = cv2.imread(fname)

    

        # Crop the image to the height

        h, w, c = img.shape

        if w > h:

            wc = int(w/2)

            w0 = wc - int(h/2)

            w1 = w0 + h

            img = img[:, w0:w1, :]

        # Rescale to N x N

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # Save

        new_fname = dest_dir + file + '.png'

        cv2.imwrite(new_fname, img)

    except:

        # Display the image name having troubles

        print(fname)

         

    # Print the progress for every N images

    if (i % 500 == 0) & (i > 0):

        print('{:} images resized in {:.2f} seconds.'.format(i, time.time()-start))



# End timing

print('Total elapsed time {:.2f} seconds.'.format(time.time()-start))
# Show rows x cols of randomly selected images

rows = 5

cols = 5

f, ax = plt.subplots(rows, cols, figsize = (rows*3, cols*3), squeeze = True)

L = SIZE

for i in range(rows*cols):

    n = np.random.randint(L) # Pool from all processed images

    file = df['image'].iloc[n]

    fname = dest_dir + file + '.png'

    img = cv2.imread(fname)

    a = ax.flat[i]

    a.imshow(img)

    a.set_title(file)

plt.tight_layout()

plt.show()
from keras.applications.xception import Xception, preprocess_input

from keras.preprocessing import image

from keras.layers import GlobalAveragePooling2D

from keras.models import Model

import numpy as np



# Create the pretraining Xception model, pop out the last dense layer

base_model = Xception(input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3))

pre_model = Model(inputs = base_model.input, outputs = base_model.get_layer('avg_pool').output)
from keras.preprocessing.image import ImageDataGenerator



# Data generator, just rescale the images from 0..255 to range 0..1

generator = ImageDataGenerator(rescale = 1./255)



# Generate flow from dataframe

data_gen = generator.flow_from_dataframe(

    dataframe = df, 

    directory = dest_dir,

    has_ext = False,

    x_col = 'image', 

    y_col = 'level', 

    target_size = (IMAGE_SIZE, IMAGE_SIZE), 

    classes = [0, 1, 2, 3, 4], 

    class_mode = 'categorical', 

    batch_size = 32, 

    shuffle = True, 

    seed = 1, 

    save_to_dir = None, 

    save_prefix = '', 

    save_format = 'png', 

    subset = None, 

    interpolation = 'nearest')
# Use preprocessing model to generate labels and predict features for Dense model

labels = data_gen.classes

features = pre_model.predict_generator(data_gen, verbose = 1)

labels.shape, features.shape
# Clear the memory from the preprocessing model

del pre_model

from keras.backend import clear_session

clear_session()



# Clear the temporary directory and its contents

dest_dir = './temp/'

for file in os.listdir(dest_dir):

    file_path = os.path.join(dest_dir, file)

    try:

        if os.path.isfile(file_path):

            os.unlink(file_path)

    except Exception as e:

        print(e)

print(dest_dir, ' cleared.')

os.rmdir(dest_dir)

print(dest_dir,'Removed.')
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, GaussianNoise

from keras.callbacks import ModelCheckpoint



# Model architecture

model = Sequential([

    Dense(46, input_dim = 2048),

    Activation('relu'),

    Dropout(0.25),

    Dense(1),

    Activation('sigmoid')

])



# Optimizer, loss and metrics

model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.summary()
# Training and validation data (x) and labels (y)

x = features

y = 1*(labels > 0)

x_train = x[:SPLIT]

y_train = y[:SPLIT]

x_valid = x[SPLIT:]

y_valid = y[SPLIT:]



# Save the model giving best validation loss

cb1 = ModelCheckpoint('best_demo10.h5', 

                      monitor = 'val_loss', 

                      verbose = 0, 

                      save_best_only = True)



# Start timing

start = time.time()



# Training

print('Training the model ...')

history = model.fit(x_train, y_train, 

                    epochs = EPOCHS, 

                    batch_size = 32, 

                    validation_data = (x_valid, y_valid), 

                    verbose = 1,

                    callbacks = [cb1])
print('Done. Elapsed time {:.0f} seconds for {:} epochs.'.format(time.time() - start, EPOCHS))



# Update the initial epoch, used for saving the previous trainings, if trained several times

model.save('demo10.h5')

print('Model saved to: demo10.h5')



# Save the history

import pickle

with open('trainHistory10', 'wb') as file_pi:

    pickle.dump(history.history, file_pi)

print('History saved to: trainHistory10')
# Plot training & validation accuracy values

plt.figure()

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])



# Plot training & validation loss values

plt.figure()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])

plt.show()
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report

from sklearn.metrics import accuracy_score

import warnings



# Function to display confusion matrix, classification report and final accuracy

def display_results(m, x, y):

    # Get the true and predicted values

    y_true = y

    predict = m.predict(x)

    y_pred = 1*(predict > 0.5)



    # Calculate and print the metrics results



    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

    

        cr = classification_report(y_true, y_pred)

        print('')

        print('Classification report:')

        print(cr)

        

        cm = confusion_matrix(y_true, y_pred)

        print('Confusion matrix:')

        print(cm)

        print('')



        a = accuracy_score(y_true, y_pred)

        print('Accuracy: {:.4f}'.format(a))

        print('')
print('RESULTS FOR FINAL MODEL')

print('TRAINING SET')

display_results(model, x_train, y_train) 

print('VALIDATION SET')

display_results(model, x_valid, y_valid)
# Load the best model and show the results for that

from keras.models import load_model

best_model = load_model('best_demo10.h5')



print('RESULTS FOR BEST MODEL')

print('TRAINING SET')

display_results(best_model, x_train, y_train) 

print('VALIDATION SET')

display_results(best_model, x_valid, y_valid)