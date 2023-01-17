import cv2

import matplotlib.pyplot as plt



# Train and test directory path

train_dir = '../input/dogs-vs-cats-redux-kernels-edition/train/'

test_dir = '../input/dogs-vs-cats-redux-kernels-edition/test/'



# Wrong images in the dataset

wrong_images = ['dog.11731.jpg', 'dog.4334.jpg', 'cat.4688.jpg', 

                'cat.11222.jpg', 'cat.1450.jpg', 'cat.2159.jpg', 

                'cat.3822.jpg', 'cat.4104.jpg', 'cat.5355.jpg', 

                'cat.7194.jpg', 'cat.7920.jpg', 'cat.9250.jpg', 

                'cat.9444.jpg', 'cat.9882.jpg', 'dog.11538.jpg', 

                'dog.8507.jpg', 'cat.2939.jpg', 'cat.3216.jpg', 

                'cat.4833.jpg', 'cat.7968.jpg', 'cat.8470.jpg', 

                'dog.10161.jpg', 'dog.10190.jpg', 'dog.11186.jpg', 

                'dog.1308.jpg', 'dog.1895.jpg', 'dog.9188.jpg', 

                'cat.5351.jpg', 'cat.5418.jpg', 'cat.9171.jpg',

                'dog.10747.jpg', 'dog.2614.jpg', 'dog.4367.jpg', 

                'dog.8736.jpg', 'cat.7377.jpg', 'dog.12376.jpg', 

                'dog.1773.jpg', 'cat.10712.jpg', 'cat.11184.jpg', 

                'cat.7564.jpg', 'cat.8456.jpg', 'dog.10237.jpg', 

                'dog.1043.jpg', 'dog.1194.jpg', 'dog.5604.jpg',

                'dog.9517.jpg', 'cat.11565.jpg', 'dog.10797.jpg', 

                'dog.2877.jpg', 'dog.8898.jpg']
def plot_grid_images(images_directory, images_label, n, m):

    """

    Shows a grid of images (5x5) with their corresponding label.

    

    Args:

        images_directory (list of str): Contains the namefiles of each image

        images_label (list of str): Contains the label of each image

        n (int): Number of rows

        m (int): Number of columns

    """

    f, ax = plt.subplots(n, m, figsize = (10, 10))



    for i in range(0,n*m):

        imgBGR = cv2.imread(images_directory[i])

        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

        ax[i//5, i%5].imshow(imgRGB)

        ax[i//5, i%5].axis('off')

        ax[i//5, i%5].set_title("{}".format(images_label[i]))



images_directory = [train_dir + image for image in wrong_images]

plot_grid_images(images_directory, wrong_images, 5, 5)

import os

import pandas as pd



# Extract the path to each image from directory

X_train = ['../input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) 

           for i in sorted(os.listdir(train_dir)) 

           if i not in wrong_images and i != 'train']



# Set 0 for cat and 1 to dog

y_train = ['cat' if 'cat.' in i else 'dog' for i in X_train]

# Note: Keep '.cat' instead of 'cat' for proper functioning of os

X_test = ['../input/dogs-vs-cats-redux-kernels-edition/test/{}'.format(i) 

          for i in sorted(os.listdir(test_dir)) if i != 'test']





# Create dataframes

train_df = pd.DataFrame({'filename':X_train, 'class':y_train})

test_df = pd.DataFrame({'filename':X_test})
# Define the sizes of each set

NUM_VALIDATION = 500

NUM_TRAIN = len(train_df) - NUM_VALIDATION

NUM_TEST = len(test_df)



# Split to train/validation/test

train_df = train_df[:NUM_TRAIN+NUM_VALIDATION].sample(frac=1)

validation_df = train_df[NUM_TRAIN:NUM_TRAIN+NUM_VALIDATION]

train_df = train_df[:NUM_TRAIN]

test_df = test_df[:NUM_TEST]
# Data augmentation specifications

HORIZONTAL_FLIP = True

WIDTH_SHIFT_RANGE = 0.2

HEIGHT_SHIFT_RANGE = 0.2

ROTATION_RANGE = 40



# Training specifications

NUM_EPOCHS = 10

BATCH_SIZE = 32 

STEPS_PER_EPOCH_TRAINING = NUM_TRAIN // BATCH_SIZE

STEPS_PER_EPOCH_VALIDATION = NUM_VALIDATION // BATCH_SIZE
from tensorflow.python import keras

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.applications.resnet50 import preprocess_input



# Image size for ResNET

IMG_SIZE = 224



# Create image generators to preprocess and group images into training 

# and validation

data_gen_train = ImageDataGenerator(preprocessing_function=preprocess_input, 

                                    horizontal_flip=HORIZONTAL_FLIP, 

                                    rotation_range=ROTATION_RANGE, 

                                    width_shift_range=WIDTH_SHIFT_RANGE, 

                                    height_shift_range=HEIGHT_SHIFT_RANGE)

data_gen_val = ImageDataGenerator(preprocessing_function=preprocess_input)



# Use .flow_from_dataframe since images are not in subfolders. 

# Otherwise use .flow_from_subfolder

train_gen = data_gen_train.flow_from_dataframe(dataframe=train_df, 

                                               x_col='filename', 

                                               y_col='class', 

                                               target_size=(IMG_SIZE, 

                                                            IMG_SIZE), 

                                               batch_size=BATCH_SIZE, 

                                               class_mode='categorical')

val_gen = data_gen_val.flow_from_dataframe(dataframe=validation_df, 

                                           x_col='filename', 

                                           y_col='class', 

                                           target_size=(IMG_SIZE, 

                                                        IMG_SIZE), 

                                           batch_size=BATCH_SIZE, 

                                           class_mode='categorical')
from tensorflow.python.keras.applications.resnet50 import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense

from tensorflow.python.keras import optimizers



# ResNet50 specifications

NUM_CLASSES = 2

POOLING = 'avg'

LAYER_ACTIVATION = 'softmax'

LOSS = 'categorical_crossentropy'

LOSS_METRICS = ['accuracy']



# Create sequential model

model = Sequential()



# First layer: ResNet50. If not specified, weights are retrieved directly 

# from the repository (Internet must be ON)

model.add(ResNet50(include_top=False, pooling=POOLING))



# Second layer: Bottleneck layer

model.add(Dense(128, activation=LAYER_ACTIVATION))



# Third layer: Dense layer for the two different classes

model.add(Dense(NUM_CLASSES, activation=LAYER_ACTIVATION))



# Provided the ResNet50 has been already trained, we are interested in tuning 

# the weights of the last layer (classification). Therefore, we keep them 

# fixed

model.layers[0].trainable = False



# Specification of optimizer, compilation of the model and summary

adam = optimizers.adam(lr=0.01)

model.compile(optimizer=adam, loss=LOSS, metrics=LOSS_METRICS)

model.summary()
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint



# Early stopping + checkpoints

EARLY_STOPPING_PATIENCE = 3 # has to be smaller than NUM_EPOCHS

cb_early_stopper = EarlyStopping(monitor='val_loss', 

                                 patience=EARLY_STOPPING_PATIENCE)

cb_checkpointer = ModelCheckpoint(filepath='../working/best.hdf5', 

                                  monitor='val_loss', 

                                  save_best_only=True,

                                  mode='auto')

# Validate the model

fit_history = model.fit_generator(

        train_gen,

        steps_per_epoch=STEPS_PER_EPOCH_TRAINING,

        epochs=NUM_EPOCHS,

        validation_data=val_gen,

        validation_steps=STEPS_PER_EPOCH_VALIDATION,

        callbacks=[cb_checkpointer, cb_early_stopper] 

)

model.load_weights('../working/best.hdf5')



# Print the different keys from the history fit

print(fit_history.history.keys())
import seaborn as sns



# Plot the responses for different events and regions

sns.set(style="darkgrid")



plt.figure(figsize=(14,3.5))

ax1 = plt.subplot(121)

sns.lineplot(x=range(len(fit_history.history['acc'])), 

             y=fit_history.history['acc'], label='acc')

sns.lineplot(x=range(len(fit_history.history['val_acc'])), 

             y=fit_history.history['val_acc'], label='val_acc')

ax1.set(xlabel='epochs', ylabel='accuracy')

ax1.set_ylim(0.9,1)

ax1.set_title('Accuracy')



ax2 = plt.subplot(122)

sns.lineplot(x=range(len(fit_history.history['loss'])), 

             y=fit_history.history['loss'], label='loss')

sns.lineplot(x=range(len(fit_history.history['val_loss'])), 

             y=fit_history.history['val_loss'], label='val_loss')

ax2.set(xlabel='epochs', ylabel='loss')

ax2.set_title('Loss');



# Note: x is set to be range(len(it_history.history['val_acc'])) and not range(NUM_EPOCHS) since early stopping
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix



# Validation without shuffle

val_gen = data_gen_val.flow_from_dataframe(dataframe=validation_df,

                                           x_col='filename',

                                           y_col='class',

                                           target_size=(IMG_SIZE, 

                                                        IMG_SIZE),

                                           batch_size=BATCH_SIZE,

                                           class_mode='categorical',

                                           shuffle=False,

                                           seed=1234)

# For reproducibility

val_gen.reset()



# Predict the validation set

y_pred = np.argmax(model.predict(val_gen), axis=1)



# Confusion Matrix

sns.heatmap(confusion_matrix(val_gen.classes, y_pred), 

            cmap='RdBu', annot=True, annot_kws={"size": 15}, 

            fmt = '.0f', xticklabels=['cats', 'dogs'], 

            yticklabels=['cats', 'dogs']).set_title("Confusion Matrix", fontsize=18);
print('Classification Report')

print(classification_report(val_gen.classes, 

                            y_pred, target_names=['Dogs', 'Cats']))
# Full batch of testing

BATCH_SIZE_TESTING = 1



# Image generator for full dataset and the testing images (not labeled)

data_gen_test = ImageDataGenerator(preprocessing_function=preprocess_input)

test_gen = data_gen_test.flow_from_dataframe(dataframe=test_df,

                                             x_col='filename',

                                             class_mode = None,

                                             target_size=(IMG_SIZE,

                                                          IMG_SIZE),

                                             batch_size=BATCH_SIZE_TESTING,

                                             shuffle=False,

                                             seed=1234)



# Each time this cell is run, we start again to go through the images

test_gen.reset()



# Predict the class of test images

prediction = model.predict(test_gen, steps=len(test_gen), verbose=1)

predicted_class_indices = np.argmax(prediction, axis=1)
predicted_labels = ['dog' if i == 1 else 'cat' 

                    for i in predicted_class_indices]

plot_grid_images(test_gen.filenames, predicted_labels, 5, 5)
# Retrieve names and probabilities

filenames = test_gen.filenames

probabilities = prediction



# Format according to competition submission requirements

label = [int(filenames[i].split('/')[-1].split('.')[0]) 

         for i in range(len(probabilities))]

prob = [probabilities[i][1] for i in range(len(probabilities))]



# Save

output = pd.DataFrame({'id': label,

                       'label': prob})

output.to_csv('submission.csv', index=False)