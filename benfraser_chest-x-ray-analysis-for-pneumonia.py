import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import pickle

import plotly.express as px

import seaborn as sns

import shutil



from keras.callbacks import ModelCheckpoint

from keras.layers import MaxPooling2D, Conv2D, Dense, Flatten, Dropout, GlobalAveragePooling2D

from keras.models import Sequential, load_model

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image



from pathlib import Path

from PIL.ExifTags import TAGS, GPSTAGS

from PIL import Image



from skimage.feature import hog

from skimage.io import imread, imshow

from skimage.transform import resize

from skimage import exposure



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix



import tensorflow as tf

import keras.backend as keras
np.random.seed(1)

tf.random.set_seed(1)
# create directory paths

old_base_dir = os.path.join('/kaggle/input/chest-xray-pneumonia', 'chest_xray')

old_train_dir = os.path.join(old_base_dir, 'train')

old_val_dir = os.path.join(old_base_dir, 'val')

old_test_dir = os.path.join(old_base_dir, 'test')
def count_data(base_dir, directories):

    """ Count number of files in selected sub-dirs, where directories 

        is a list of strings for each sub-directory path. Also returns a 

        dictionary of all img paths (values) for each sub-dir (keys).

    """

    

    # list to store img counts, and dict to store image paths

    file_counts = []

    img_paths = {}

    

    for directory in directories:

        

        img_files = [x for x in os.listdir(os.path.join(base_dir, directory)) 

                 if x.endswith('.jpeg')]

        

        # find paths to all imgs

        path_names = [os.path.join(base_dir, directory, x) for x in img_files]

        

        # count img no. and append to file counts

        num_files = len(img_files)

        file_counts.append(num_files)

    

        # update dict of paths with the given imgs for the sub-dir

        key_name = directory.replace('/', '_').lower()

        img_paths[key_name] = path_names

    

    return file_counts, img_paths
split_type = ['train', 'val', 'test']

class_type = ['PNEUMONIA', 'NORMAL']

directories = [f"{x}/{y}" for x in split_type for y in class_type]



counts, img_paths = count_data(old_base_dir, directories)



for subdir, count in zip(directories, counts):

    print(f"{subdir} : {count}")

    

sns.barplot(y=directories, x=counts)

plt.show()
# create new directory paths

base_dir = os.path.join('/kaggle/working', 'chest_xray')

train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'val')

test_dir = os.path.join(base_dir, 'test')
%%time



# iterate through each sub-dir - count imgs and form dict of paths

for directory in directories:

    

    # create new directory structure in kaggle working dir

    new_dir = os.path.join(base_dir, directory)

    Path(new_dir).mkdir(parents=True, exist_ok=True)

    

    # gather img files in kaggle read-only dir

    img_files = [x for x in os.listdir(os.path.join(old_base_dir, directory)) 

                 if x.endswith('.jpeg')]

    

    # find paths to old and new paths for images in current directory

    old_path_names = [os.path.join(old_base_dir, directory, x) for x in img_files]

    new_path_names = [os.path.join(new_dir, x) for x in img_files]

    

    print(f"Moving and resizing directory: {directory}\n")

    

    for i in range(len(old_path_names)):

        

        # load img, resize, and save to new location

        img = Image.open(old_path_names[i])

        img_new = img.resize((360,320), Image.ANTIALIAS)

        img_new.save(new_path_names[i], 'JPEG', quality=90)
def move_img_data(source_dir, destination_dir, proportion=0.2, suffix='.jpeg'):

    """ Move a random proportion of img data from a source to destination directory """

    

    img_files = [x for x in os.listdir(source_dir) if x.endswith(suffix)]

    

    move_num = int(np.ceil(len(img_files)*proportion))

    

    # select random proportion of images to move

    random_indices = np.random.permutation(len(img_files))[:move_num]

    

    print(f"Moving a total of {move_num} images from "

          f"{source_dir} to {destination_dir}\n")

    

    # move selected images to destination loc

    for index in random_indices:

        src_path = os.path.join(source_dir, img_files[index])

        dest_path = os.path.join(destination_dir, img_files[index])

        shutil.copyfile(src_path, dest_path)
# move 20% of training samples from train to val dir for both classes - ONLY RUN ONCE

move_img_data(os.path.join(train_dir, 'NORMAL'), 

              os.path.join(validation_dir, 'NORMAL'),

              proportion=0.2)

move_img_data(os.path.join(train_dir, 'PNEUMONIA'), 

              os.path.join(validation_dir, 'PNEUMONIA'),

              proportion=0.2)
counts, img_paths = count_data(base_dir, directories)



for subdir, count in zip(directories, counts):

    print(f"{subdir} : {count}")

    

sns.barplot(y=directories, x=counts)

plt.show()
def create_dataframe(data_dir):

    """ Returns a dataframe consisting of img path and label, where

        0 is normal and 1 is pneumonia """

    data = []

    labels = []

    

    # obtain image paths for all training data

    normal_dir = os.path.join(data_dir, 'NORMAL')

    pneunomia_dir = os.path.join(data_dir, 'PNEUMONIA')

    normal_data = [x for x in os.listdir(normal_dir) if x.endswith('.jpeg')]

    pneunomia_data = [x for x in os.listdir(pneunomia_dir) if x.endswith('.jpeg')]

    

    # append img path and labels for each

    for normal in normal_data:

        data.append(os.path.join(normal_dir, normal))

        labels.append(0) 

    for pneumonia in pneunomia_data:

        data.append(os.path.join(pneunomia_dir, pneunomia_dir))

        labels.append(1)

        

    # return pandas dataframe

    return pd.DataFrame({'Image_path' : data, 'Label' : labels})
train_df = create_dataframe(train_dir)

val_df = create_dataframe(validation_dir)

test_df = create_dataframe(test_dir)
train_df['Label'].value_counts().plot.bar()

plt.show()
def duplicate_data(file_dir, suffix='.jpeg'):

    """ duplicate img data within destination directory """

    

    img_files = [x for x in os.listdir(file_dir) if x.endswith(suffix)]

    

    for img in img_files:

        src_path = os.path.join(file_dir, img)

        dup_img = f"{img[:-len(suffix)]}_2{suffix}"

        dest_path = os.path.join(file_dir, dup_img)

        shutil.copyfile(src_path, dest_path)
duplicate_data(os.path.join(train_dir, 'NORMAL'))
train_df = create_dataframe(train_dir)

train_df['Label'].value_counts().plot.bar()

plt.show()
fig = plt.figure(figsize=(12, 6))



for i, example in enumerate(img_paths['train_pneumonia'][:5]):

    

    ax = fig.add_subplot(2, 5, i+1)

    ax.set_xticks([])

    ax.set_yticks([])

    

    # read image and plot

    example_img = tf.io.read_file(example)

    example_img = tf.image.decode_jpeg(example_img, channels=3)

    example_img = tf.image.resize(example_img, [360, 320])

    example_img /= 255.0

    ax.imshow(example_img)

    ax.set_title(f"Pneumonia {i}")

    

for i, example in enumerate(img_paths['train_normal'][:5]):

    

    ax = fig.add_subplot(2, 5, i+6)

    ax.set_xticks([])

    ax.set_yticks([])

    

    # read image and plot

    example_img = tf.io.read_file(example)

    example_img = tf.image.decode_jpeg(example_img, channels=3)

    example_img = tf.image.resize(example_img, [360, 320])

    example_img /= 255.0

    ax.imshow(example_img)

    ax.set_title(f"Normal {i}")
img_height, img_width = 150, 150

batch_size = 10



# training data augmentation - rotate, shear, zoom and flip

train_datagen = ImageDataGenerator(

    rotation_range = 30,

    rescale = 1.0 / 255.0,

    shear_range = 0.2,

    zoom_range = 0.2,

    horizontal_flip = True,

    vertical_flip=True)



# no augmentation for test data - only rescale

test_datagen = ImageDataGenerator(rescale = 1. / 255.0)



# generate batches of augmented data from training data

train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='binary')



# generate val data from val dir

validation_generator = test_datagen.flow_from_directory(

    validation_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='binary')



nb_train_samples = len(train_generator.classes)

nb_validation_samples = len(validation_generator.classes)
# get class labels dict containing index of each class for decoding predictions

class_labels = train_generator.class_indices



class_labels
def create_CNN(input_size=(150, 150)):

    """ Basic CNN with 4 Conv layers, each followed by a max pooling """

    cnn_model = Sequential()

    

    # four Conv layers with max pooling

    cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

    cnn_model.add(MaxPooling2D(2, 2))

    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))

    cnn_model.add(MaxPooling2D(2, 2))

    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))

    cnn_model.add(MaxPooling2D(2, 2))

    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))

    cnn_model.add(MaxPooling2D(2, 2))

    

    # flatten output and feed to dense layer, via dropout layer

    cnn_model.add(Flatten())

    cnn_model.add(Dropout(0.5))

    cnn_model.add(Dense(512, activation='relu'))

    

    # add output layer - sigmoid since we only have 2 outputs

    cnn_model.add(Dense(1, activation='sigmoid'))

    

    cnn_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    

    return cnn_model
CNN_model = create_CNN()

CNN_model.summary()
# set up a check point for our model - save only the best val performance

save_path ="basic_cnn_best_weights.hdf5"



trg_checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', 

                                 verbose=1, save_best_only=True, mode='max')



trg_callbacks = [trg_checkpoint]
# batch steps before an epoch is considered complete (trg_size / batch_size):

steps_per_epoch = np.ceil(nb_train_samples/batch_size)



# validation batch steps (val_size / batch_size):

val_steps_per_epoch = np.ceil(nb_validation_samples/batch_size)
history = CNN_model.fit(train_generator, epochs=50, 

                        steps_per_epoch=steps_per_epoch, 

                        validation_data=validation_generator, 

                        validation_steps=val_steps_per_epoch,

                        callbacks=trg_callbacks,

                        shuffle=True)
# save model as a HDF5 file with weights + architecture

CNN_model.save('basic_cnn_model_1.hdf5')



# save the history of training to a datafile for later retrieval

with open('history_basic_cnn_model_1.pickle', 

          'wb') as pickle_file:

    pickle.dump(history.history, pickle_file)

    

loaded_model = False
# if already trained - import history file and best training weights

CNN_model = load_model('basic_cnn_best_weights.hdf5')
# if already trained - import history file and training weights

#CNN_model = load_model('models/basic_cnn_model_1.hdf5')



# get history of trained model

#with open('models/history_basic_cnn_model_1.pickle', 'rb') as handle:

#    history = pickle.load(handle)

    

#loaded_model = True
# if loaded model set history accordingly

if loaded_model:

    trg_hist = history

else:

    trg_hist = history.history



trg_loss = trg_hist['loss']

val_loss = trg_hist['val_loss']



trg_acc = trg_hist['accuracy']

val_acc = trg_hist['val_accuracy']



epochs = range(1, len(trg_acc) + 1)



# plot losses and accuracies for training and validation 

fig = plt.figure(figsize=(12,6))

ax = fig.add_subplot(1, 2, 1)

plt.plot(epochs, trg_loss, marker='o', label='Training Loss')

plt.plot(epochs, val_loss, marker='x', label='Validation Loss')

plt.title("Training / Validation Loss")

ax.set_ylabel("Loss")

ax.set_xlabel("Epochs")

plt.legend(loc='best')



ax = fig.add_subplot(1, 2, 2)

plt.plot(epochs, trg_acc, marker='o', label='Training Accuracy')

plt.plot(epochs, val_acc, marker='^', label='Validation Accuracy')

plt.title("Training / Validation Accuracy")

ax.set_ylabel("Accuracy")

ax.set_xlabel("Epochs")

plt.legend(loc='best')

plt.tight_layout()

plt.show()
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_height, img_width), 

                                                  batch_size=4, class_mode='binary')



test_loss, test_accuracy = CNN_model.evaluate_generator(test_generator)

print(f"Test accuracy: {test_accuracy}")
# generate val data from val dir

test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='binary',

    shuffle=False)
y_labels = np.expand_dims(test_generator.classes, axis=1)

nb_test_samples = len(y_labels)



y_preds = CNN_model.predict(test_generator, 

                            steps=np.ceil(nb_test_samples/batch_size))



# round predictions to 0 (Normal) or 1 (Pneumonia)

y_preds = np.rint(y_preds)
# number of incorrect labels

incorrect = (y_labels[:, 0] != y_preds[:, 0]).sum()



# print the basic results of the model

print(f"Accuracy: {accuracy_score(y_labels[:, 0], y_preds[:, 0])*100:.2f}%")

print(f"F1 Score: {f1_score(y_labels[:, 0], y_preds[:, 0]):.2f}")

print(f"Samples incorrectly classified: {incorrect} out of {len(y_labels)}")
# print recall, precision and f1 score results

print(classification_report(y_labels[:, 0], y_preds[:, 0]))
def plot_confusion_matrix(true_y, pred_y, title='Confusion Matrix', figsize=(8,6)):

    """ Custom function for plotting a confusion matrix for predicted results """

    conf_matrix = confusion_matrix(true_y, pred_y)

    conf_df = pd.DataFrame(conf_matrix, columns=np.unique(true_y), index = np.unique(true_y))

    conf_df.index.name = 'Actual'

    conf_df.columns.name = 'Predicted'

    plt.figure(figsize = figsize)

    plt.title(title)

    sns.set(font_scale=1.4)

    sns.heatmap(conf_df, cmap="Blues", annot=True, 

                annot_kws={"size": 16}, fmt='g')

    plt.show()

    return





# plot a confusion matrix of our results

plot_confusion_matrix(y_labels[:, 0], y_preds[:, 0], title="Basic ConvNet Confusion Matrix")
# get class labels dict containing index of each class for decoding predictions

class_labels = train_generator.class_indices



# obtain a reverse dict to convert index into class labels

reverse_class_index = {i : class_label for class_label, i in class_labels.items()}
class_labels
def process_and_predict_img(image_path, model, img_size=(150, 150)):

    """ Utility function for making predictions for an image. """

    img_path = image_path

    img = image.load_img(img_path, target_size=img_size)

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = test_datagen.standardize(x)

    predictions = model.predict(x)

    return img, predictions
img, prediction = process_and_predict_img(img_paths['test_normal'][0], 

                                          model=CNN_model)

plt.imshow(img)

plt.title(f"Prediction: {reverse_class_index[np.argmax(prediction)]}\n")

plt.show()
def f1_score(y_true, y_pred):

    """ Find and return the F1 Score """

    y_pred = keras.round(y_pred)

    

    # calculate true pos, true neg, false pos, false neg

    true_pos = keras.sum(keras.cast(y_true*y_pred, 'float'), axis=0)

    true_neg = keras.sum(keras.cast((1 - y_true)*(1 - y_pred), 'float'), axis=0)

    false_pos = keras.sum(keras.cast((1- y_true)*y_pred, 'float'), axis=0)

    false_neg = keras.sum(keras.cast(y_true*(1 - y_pred), 'float'), axis=0)



    # calculate precision / recall, adding epsilon to prevent zero div error(s)

    precision = true_pos / (true_pos + false_pos + keras.epsilon())

    recall = true_pos / (true_pos + false_neg + keras.epsilon())



    # calculate f1 score and return

    f1_score = (2.0 * precision * recall) / (precision + recall + keras.epsilon())

    f1_score = tf.where(tf.math.is_nan(f1_score), tf.zeros_like(f1_score), f1_score)

    return keras.mean(f1_score)





def f1_loss(y_true, y_pred):

    """ Calculate mean F1 and return minimising function to approximate a loss equivalent. """

    return 1 - f1_score(y_true, y_pred)
def basic_CNN_2(input_size=(150, 150)):

    """ Basic CNN with 4 Conv and max pooling layers, with custom F1 loss """

    cnn_model = Sequential()

    

    # four Conv layers with max pooling

    cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

    cnn_model.add(MaxPooling2D(2, 2))

    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))

    cnn_model.add(MaxPooling2D(2, 2))

    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))

    cnn_model.add(MaxPooling2D(2, 2))

    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))

    cnn_model.add(MaxPooling2D(2, 2))

    

    # flatten output and feed to dense layer, via dropout layer

    cnn_model.add(Flatten())

    cnn_model.add(Dropout(0.5))

    cnn_model.add(Dense(512, activation='relu'))

    

    # add output layer - sigmoid since we only have 2 outputs

    cnn_model.add(Dense(1, activation='sigmoid'))

    

    cnn_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', f1_score])

    

    return cnn_model
CNN_model_2 = basic_CNN_2()

CNN_model_2.summary()
# set up a check point for our model - save only the best val performance

save_path ="basic_cnn_2_best_weights.hdf5"



trg_checkpoint = ModelCheckpoint(save_path, monitor='val_f1_score', 

                                 verbose=1, save_best_only=True, mode='max')



trg_callbacks = [trg_checkpoint]



# batch steps before an epoch is considered complete (trg_size / batch_size):

steps_per_epoch = np.ceil(nb_train_samples/batch_size)

val_steps_per_epoch = np.ceil(nb_validation_samples/batch_size)
history = CNN_model_2.fit(train_generator, epochs=50, 

                          steps_per_epoch=steps_per_epoch, 

                          validation_data=validation_generator, 

                          validation_steps=val_steps_per_epoch,

                          callbacks=trg_callbacks,

                          shuffle=True)
# save model as a HDF5 file with weights + architecture

CNN_model_2.save('basic_cnn_model_2.hdf5')



# save the history of training to a datafile for later retrieval

with open('history_basic_cnn_model_2.pickle', 

          'wb') as pickle_file:

    pickle.dump(history.history, pickle_file)

    

loaded_model = False
# if already trained - import history file and best training weights

CNN_model_2 = load_model('basic_cnn_2_best_weights.hdf5', custom_objects={'f1_score' : f1_score})
# get history of trained model

#with open('history_basic_cnn_model_2.pickle', 'rb') as handle:

#    history = pickle.load(handle)

    

#loaded_model = True
# if loaded model set history accordingly

if loaded_model:

    trg_hist = history

else:

    trg_hist = history.history



trg_loss = trg_hist['loss']

val_loss = trg_hist['val_loss']



trg_acc = trg_hist['accuracy']

val_acc = trg_hist['val_accuracy']



epochs = range(1, len(trg_acc) + 1)



trg_loss = trg_hist['loss']

val_loss = trg_hist['val_loss']



trg_acc = trg_hist['accuracy']

val_acc = trg_hist['val_accuracy']



trg_f1 = trg_hist['f1_score']

val_f1 = trg_hist['val_f1_score']



epochs = range(1, len(trg_acc) + 1)



# plot losses and accuracies for training and validation 

fig = plt.figure(figsize=(15,5))

ax = fig.add_subplot(1, 2, 1)

plt.plot(epochs, trg_loss, marker='o', label='Training Loss')

plt.plot(epochs, val_loss, marker='x', label='Validation Loss')

plt.title("Training / Validation Loss")

ax.set_ylabel("Loss")

ax.set_xlabel("Epochs")

plt.legend(loc='best')



ax = fig.add_subplot(1, 2, 2)

plt.plot(epochs, trg_acc, marker='o', label='Training Accuracy')

plt.plot(epochs, val_acc, marker='^', label='Validation Accuracy')

plt.title("Training / Validation Accuracy")

ax.set_ylabel("Accuracy")

ax.set_xlabel("Epochs")

plt.legend(loc='best')

plt.tight_layout()

plt.show()



# plot F1 scores

fig = plt.figure(figsize=(7,5))

ax = fig.add_subplot(1, 1, 1)

plt.plot(epochs, trg_acc, marker='o', label='Training F1')

plt.plot(epochs, val_acc, marker='^', label='Validation F1')

plt.title("Training / Validation F1 Score")

ax.set_ylabel("F1 Score")

ax.set_xlabel("Epochs")

plt.legend(loc='best')

plt.tight_layout()

plt.show()
# generate val data from val dir

test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='binary',

    shuffle=False)



y_labels = np.expand_dims(test_generator.classes, axis=1)

nb_test_samples = len(y_labels)
y_preds = CNN_model_2.predict(test_generator, 

                              steps=np.ceil(nb_test_samples/batch_size))



# round predictions to 0 (Normal) or 1 (Pneumonia)

y_preds = np.rint(y_preds)
# number of incorrect labels

incorrect = (y_labels[:, 0] != y_preds[:, 0]).sum()



# print the basic results of the model

print(f"Accuracy: {accuracy_score(y_labels[:, 0], y_preds[:, 0])*100:.2f}%")

print(f"F1 Score: {f1_score(y_labels[:, 0], y_preds[:, 0]):.2f}")

print(f"Samples incorrectly classified: {incorrect} out of {len(y_labels)}")
# print recall, precision and f1 score results

print(classification_report(y_labels[:, 0], y_preds[:, 0]))
def plot_confusion_matrix(true_y, pred_y, title='Confusion Matrix', figsize=(8,6)):

    """ Custom function for plotting a confusion matrix for predicted results """

    conf_matrix = confusion_matrix(true_y, pred_y)

    conf_df = pd.DataFrame(conf_matrix, columns=np.unique(true_y), index = np.unique(true_y))

    conf_df.index.name = 'Actual'

    conf_df.columns.name = 'Predicted'

    plt.figure(figsize = figsize)

    plt.title(title)

    sns.set(font_scale=1.4)

    sns.heatmap(conf_df, cmap="Blues", annot=True, 

                annot_kws={"size": 16}, fmt='g')

    plt.show()

    return





# plot a confusion matrix of our results

plot_confusion_matrix(y_labels[:, 0], y_preds[:, 0], title="ConvNet (F1 loss) Confusion Matrix")
try:

    shutil.rmtree(base_dir)

except OSError as e:

    print("Error: %s : %s" % (base_dir, e.strerror))