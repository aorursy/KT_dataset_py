import os

import glob

import warnings

import cv2



from pathlib import Path



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import classification_report, confusion_matrix



from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator



from keras.applications import VGG16

from keras.applications import inception_v3

from keras.applications.vgg16 import preprocess_input

from keras.models import Sequential, Model

from keras.models import InputLayer

from keras.models import load_model

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from keras.optimizers import SGD



import tensorflow as tf



%matplotlib inline



warnings.filterwarnings("ignore")



from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)
# set data path



data_path = "/kaggle/input/intel-image-classification/"

data_path_train = data_path + "seg_train/seg_train/"

data_path_validation = data_path + "seg_test/seg_test/"

data_path_prediction = data_path + "seg_pred/seg_pred/"

# get all image file path

file_names = [(os.path.abspath(x), os.path.relpath(x, data_path), os.path.basename(x)) 

              for x in Path(data_path).glob("**/*.jpg")]



# store image meta-data into dataframe

df = pd.DataFrame(

    file_names, 

    columns=["full_file_name", "rel_file_name", "file_name"])



# set image data type

df["image_type"] = df['rel_file_name'].map(

    lambda x: 

        "pred" if "pred" in x 

        else x.split("/")[0].replace("seg_", ""))



# set image classification

df["image_classification"] = df['rel_file_name'].map(

    lambda x: 

        "?" if "pred" in x 

        else x.split("/")[2].replace("seg_", ""))



df.sample(5)

# visualize it



fig, ax = plt.subplots(1, 2, figsize=(15, 5))

sns.color_palette("dark", 4)



palette = sns.color_palette("muted", 6)

sns.set_palette(palette)



sns.countplot(

    data = df, x = "image_type", 

    order = ["train", "test", "pred"], 

    ax = ax[0])



sns.countplot(

    data = df.loc[df["image_classification"] != "?", :], 

    x = "image_type", hue = "image_classification", 

    order = ["train", "test", "pred"], 

    hue_order = ["buildings", "forest", "glacier", "mountain", "sea", "street"],

    ax = ax[1])

# Inspect images



mosaic_size = (20, 20)

mosaic_number_of_images = 80

mosaic_number_of_columns = 8

mosaic_number_of_rows = int(mosaic_number_of_images / mosaic_number_of_columns)



mosaic_tile_text_fontColor = (0, 255, 0)

display_image_size = (768, 768, 3)



df_display_items = df.loc[df["image_type"] != "pred"].sample(mosaic_number_of_images)

mosaic_image = np.zeros(

    (mosaic_number_of_rows * (display_image_size[0] + 2), 

     mosaic_number_of_columns * (display_image_size[1] + 2), 

     3))



counter = 0

for i in range(mosaic_number_of_rows):

    

    y_start = i * (display_image_size[1] + 2)

    y_end = y_start + display_image_size[1]

    

    j = 0

    while j < mosaic_number_of_columns:

        

        x_start = j * (display_image_size[0] + 2)

        x_end = x_start + display_image_size[0]

        

        image_path = df_display_items.iloc[counter, :]["full_file_name"]

        image_classification = df_display_items.iloc[counter, :]["image_classification"]



        img = image.load_img(image_path, target_size = display_image_size)

        img = image.img_to_array(img) / 255

 

        # print category into the image itself

        cv2.putText(

            img = img, 

            text = image_classification,

            org = (10, 150),

            fontFace = cv2.FONT_HERSHEY_PLAIN,

            fontScale = 8,

            color = mosaic_tile_text_fontColor,

            thickness = 8,

            lineType = 8)

       

        mosaic_image[y_start:y_end, x_start:x_end, :] = img[:,:,:]



        j += 1

        counter += 1



plt.figure(figsize = mosaic_size)

plt.imshow(mosaic_image[:, :, :], cmap='seismic')
parameter_batch_size_train = 250

parameter_batch_size_val = 250



parameter_image_data_rotation_range = 0.4

parameter_image_data_horizontal_flip = True



img_height = 299

img_width = 299

img_channels = 3

# contruct train image data generator



train_datagen = ImageDataGenerator(

    rotation_range = parameter_image_data_rotation_range,

    horizontal_flip = parameter_image_data_horizontal_flip,

    preprocessing_function = inception_v3.preprocess_input)  



X_train_gen = train_datagen.flow_from_directory(

    data_path_train,

    target_size = (img_height, img_width),

    batch_size = parameter_batch_size_train)

# contruct validation image data generator



val_datagen = ImageDataGenerator(

    preprocessing_function = inception_v3.preprocess_input)



X_val_gen = val_datagen.flow_from_directory(

    data_path_validation,

    target_size = (img_height, img_width),

    batch_size = parameter_batch_size_val)

# get Google's inception V3 as pre-trainned model



model_base = inception_v3.InceptionV3(

    input_shape = (img_height, img_height, img_channels), 

    include_top = False)



#model_base.summary()
# build a new model with the featurizer and a new classifier



model_classifier = Sequential()



model_classifier.add(InputLayer(input_shape=(8, 8, 2048))) # InceptionV3

model_classifier.add(Flatten())

model_classifier.add(Dense(32, activation='relu'))

model_classifier.add(Dropout(0.5))

model_classifier.add(Dense(6, activation='softmax'))



model_classifier.summary()
# build a combined model, consists of incepction V3 as base model and our new classifier model



model_classifier_output = model_classifier(model_base.output) # classifier consumes the output of the CNN



model_combined = Model(

    inputs  = model_base.input,      # CNN without classifier

    outputs = model_classifier_output)     # Classifier



model_combined.summary()
# Gets all layers except classifier (the last one)

# Freeze layers up to the classifier

# This will preserve the weights for layers during backpropagation



for layer in model_combined.layers[:-1]:

    layer.trainable = False # freeze



#model_combined.summary()
parameter_cnn_learning_rate = 1e-3

parameter_cnn_epochs = 20

parameter_cnn_steps_per_epoch = 20

parameter_cnn_validation_steps = 20

sgd = SGD(lr = parameter_cnn_learning_rate)



model_combined.compile(

    optimizer = sgd,                   

    loss = 'categorical_crossentropy',

    metrics=['acc'])
import time

tb = TensorBoard(log_dir='logs'.format(int(time.time())))

es = EarlyStopping(patience=3)

mc = ModelCheckpoint('transfer.{epoch:02d}-{val_loss:.2f}.hdf5')



tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")

#%load_ext tensorboard.notebook

#%tensorboard --logdir logs



history = model_combined.fit_generator(

    X_train_gen,

    steps_per_epoch = parameter_cnn_steps_per_epoch,

    epochs = parameter_cnn_epochs,

    validation_data = X_val_gen,

    validation_steps = parameter_cnn_validation_steps)
#model_path = 'transfer.02-0.51.hdf5' # update model path

#best_model = load_model(model_path)



best_model = model_combined
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))



ax[0].plot(history.history['acc'], label='train')

ax[0].plot(history.history['val_acc'], label='val')

ax[0].set_ylabel('accuracy')

ax[0].set_xlabel('epochs')

ax[0].legend()



ax[1].plot(history.history['loss'], label='train')

ax[1].plot(history.history['val_loss'], label='val')

ax[1].set_ylabel('loss')

ax[1].set_xlabel('epochs')

ax[1].legend()



plt.show()
X_val, y_val = X_val_gen.next()

y_val_classes = y_val.argmax(axis=1)
# probabilities

pred = best_model.predict(X_val)



# probabilities to classes

pred_classes = pred.argmax(axis=1)

#pred_classes
print(classification_report(y_val_classes, pred_classes))
print(confusion_matrix(y_val_classes, pred_classes))
labels = X_val_gen.class_indices

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in pred_classes]



labels

#predictions
# fully adapted 

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

# https://www.kaggle.com/grfiv4/plot-a-confusion-matrix

    

import itertools



def plot_confusion_matrix(

    cm,

    target_names,

    title = 'Confusion matrix',

    cmap = None,

    normalize = True):



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(5, 5))

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)

    plt.title(title)



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.2f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))



    plt.show()
plot_confusion_matrix(

    cm = confusion_matrix(y_val_classes, pred_classes), 

    normalize = True,

    target_names = labels.items(),

    title = "Normalized Confusion Matrix")
df_view = df.loc[df["image_type"] == "test"]



df_view["image_classification_predict"] = ""



for index, row in df_view.iterrows():

   

    img = image.load_img(row["full_file_name"], target_size=(img_height, img_width)) 

    

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = inception_v3.preprocess_input(x)    

    

    predict = best_model.predict(x).argmax(axis=1)

    predict_decode = [labels[k] for k in predict]    

    

    row["image_classification_predict"] = predict_decode[0]

    
df_show = df_view.loc[df_view["image_classification_predict"] != df_view["image_classification"], :]



df_show.shape

df_view.shape


# Inspect images # TODO REFACTOR THIS

mosaic_size = (20, 20)

mosaic_number_of_images = 80

mosaic_number_of_columns = 8

mosaic_number_of_rows = int(mosaic_number_of_images / mosaic_number_of_columns)

mosaic_tile_text_fontColor = (0, 255, 0)

display_image_size = (768, 768, 3)

df_display_items = df_show.sample(mosaic_number_of_images)

mosaic_image = np.zeros(

    (mosaic_number_of_rows * (display_image_size[0] + 2), 

     mosaic_number_of_columns * (display_image_size[1] + 2), 

     3))

counter = 0

for i in range(mosaic_number_of_rows):

    

    y_start = i * (display_image_size[1] + 2)

    y_end = y_start + display_image_size[1]

    

    j = 0

    while j < mosaic_number_of_columns:

        

        x_start = j * (display_image_size[0] + 2)

        x_end = x_start + display_image_size[0]

        

        image_path = df_display_items.iloc[counter, :]["full_file_name"]

        image_classification = df_display_items.iloc[counter, :]["image_classification"]

        image_classification_prediction = df_display_items.iloc[counter, :]["image_classification_predict"]

        img = image.load_img(image_path, target_size = display_image_size)

        img = image.img_to_array(img) / 255

 

        # print category into the image itself

        cv2.putText(

            img = img, 

            text = image_classification,

            org = (10, 150),

            fontFace = cv2.FONT_HERSHEY_PLAIN,

            fontScale = 8,

            color = mosaic_tile_text_fontColor,

            thickness = 8,

            lineType = 8)

        

        # print category into the image itself

        cv2.putText(

            img = img, 

            text = image_classification_prediction,

            org = (10, 650),

            fontFace = cv2.FONT_HERSHEY_PLAIN,

            fontScale = 8,

            color = (255, 0, 0),

            thickness = 8,

            lineType = 8)        

       

        mosaic_image[y_start:y_end, x_start:x_end, :] = img[:,:,:]

        j += 1

        counter += 1

plt.figure(figsize = mosaic_size)

plt.imshow(mosaic_image[:, :, :], cmap='seismic')
