import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os, shutil, cv2, time
SIZE = 2400 # Total number of images pooled from 300_train

SPLIT = 1800 # Number of images used for training

BATCH_SIZE = 16

TRAIN_STEPS = SPLIT // BATCH_SIZE

VALID_STEPS = (SIZE - SPLIT) // BATCH_SIZE

EPOCHS = 100

N_EPOCHS = 0

IMAGE_SIZE = 299

print('Training set:    ', SPLIT)

print('Validation set:  ', SIZE - SPLIT)

print('Batch size:      ', BATCH_SIZE)

print('Training steps:  ', TRAIN_STEPS)

print('Validation steps:', VALID_STEPS)

print('Epochs:          ', EPOCHS)

print('Image size:      ', (IMAGE_SIZE, IMAGE_SIZE, 3))
# Create destination directory

dest_dir = './temp/'



try:

    os.mkdir(dest_dir)

    print('Created a directory:', dest_dir)

except:

    # Temp directory already exist, so clear it

    for file in os.listdir(dest_dir):  

        file_path = os.path.join(dest_dir, file)

        try:

            if os.path.isfile(file_path):

                os.unlink(file_path)

        except Exception as e:

            print(e)

    print(dest_dir, ' cleared.')
# Read the train image names and labels

df = pd.read_csv('../input/newTrainLabels.csv')



# Take SIZE number of samples from dataframe

df = df.sample(n = SIZE, random_state = 1)



# Use binary labels

df['level'] = 1*(df['level'] > 0)



# Source directory

source_dir = '../input/300_train/300_train/'



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

        N = IMAGE_SIZE

        img = cv2.resize(img, (N, N))

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
from keras.preprocessing.image import ImageDataGenerator



# Validation generator, only rescale the images from 0..255 to range 0..1

valid_generator = ImageDataGenerator(rescale = 1./255)



# Train generator: zoom, rotate and flip

train_generator = ImageDataGenerator(

    rescale = 1./255,

    zoom_range = 0.1,

    rotation_range = 180,

    horizontal_flip = True,

    vertical_flip = True)



# Training flow

print('Training flow:')

train_flow = train_generator.flow_from_dataframe(

    dataframe = df[:SPLIT], # read from data frame from 0 to SPLIT

    directory = dest_dir,

    has_ext = False,

    x_col = 'image', 

    y_col = 'level', 

    target_size = (IMAGE_SIZE, IMAGE_SIZE), 

    classes = [0, 1], 

    class_mode = 'binary', 

    batch_size = BATCH_SIZE, 

    shuffle = True, 

    seed = 1)



# Validation flow

print('Validation flow:')

valid_flow = valid_generator.flow_from_dataframe(

    dataframe = df[SPLIT:], # read data frame from SPLIT to END

    directory = dest_dir,

    has_ext = False,

    x_col = 'image', 

    y_col = 'level', 

    target_size = (IMAGE_SIZE, IMAGE_SIZE), 

    classes = [0, 1], 

    class_mode = 'binary', 

    batch_size = BATCH_SIZE, 

    shuffle = False)
from keras.applications.xception import Xception

from keras.layers import Dense, Activation

from keras.models import Model

import keras.backend as K



# Clear backend (tensorflow)

K.clear_session()



# Create the pretraining Xception model, pop out the last dense layer

base_model = Xception(input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3))

x = Dense(1)(base_model.get_layer('avg_pool').output)

y = Activation('sigmoid')(x)

model = Model(inputs = base_model.input, outputs = y)



# Optimizer, loss and metrics

model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])

#model.summary()
from keras.callbacks import ModelCheckpoint



# Save the model giving best validation loss

cb1 = ModelCheckpoint('best_demo11.h5', 

                      monitor = 'val_loss', 

                      verbose = 0, 

                      save_best_only = True)



# Start timing

start = time.time()



# Training

print('Training the model ...')

history = model.fit_generator(

    generator = train_flow,

    steps_per_epoch = TRAIN_STEPS,

    epochs = N_EPOCHS + EPOCHS,

    initial_epoch = N_EPOCHS,

    verbose = 1,

    callbacks = [cb1],

    validation_data = valid_flow,

    validation_steps = VALID_STEPS)

stop = time.time()

etime = stop - start

print('Done. Elapsed time {:.0f} seconds for {:} epochs, average {:.1f} seconds/epoch.'.format(etime, EPOCHS, etime/EPOCHS))



# Update the initial epoch, used for saving the previous trainings, if trained several times

N_EPOCHS = N_EPOCHS + EPOCHS

model.save('demo11.h5')



# Save the history

import pickle

with open('trainHistory', 'wb') as file_pi:

    pickle.dump(history.history, file_pi)
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

def display_results(m, flow):

    # Get the true and predicted values

    y_true = flow.classes

    predict = m.predict_generator(flow)

    y_pred = 1*(predict > 0.5)



    # Calculate and print the metrics results



    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

    

        cm = confusion_matrix(y_true, y_pred)

        print('Confusion matrix:')

        print(cm)

        print('')



        a = accuracy_score(y_true, y_pred)

        print('Accuracy: {:.4f}'.format(a))

        print('')



        cr = classification_report(y_true, y_pred)

        print('Classification report:')

        print(cr)
print('RESULTS FOR FINAL MODEL')

print('VALIDATION SET')

display_results(model, valid_flow)
# Load the best model and show the results for that

from keras.models import load_model



print('RESULTS FOR BEST MODEL')

print('VALIDATION SET')

best_model = load_model('best_demo11.h5')

display_results(best_model, valid_flow) 
# Clear the temporary directory

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