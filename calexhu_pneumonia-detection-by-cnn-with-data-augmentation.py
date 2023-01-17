import tensorflow as tf

tf.__version__
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mimg

import seaborn as sns

%matplotlib inline

from sklearn.metrics import confusion_matrix



import cv2

import os

import glob



from os import listdir, makedirs, getcwd, remove

from os.path import isfile, join, abspath, exists, isdir, expanduser

from PIL import Image

from pathlib import Path

from skimage.io import imread

from skimage.transform import resize



from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, SeparableConv2D

from tensorflow.keras.layers import GlobalMaxPooling2D, Flatten, Dropout

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.optimizers import Adam, RMSprop, SGD
# Input data files are available in the "../input/" directory.

INPUT_PATH = "../input/pneumonia-detection/chest_xray"



# List the files in the input directory.

print(os.listdir(INPUT_PATH))
# list of all the training images

train_normal = Path(INPUT_PATH + '/train/NORMAL').glob('*.jpeg')

train_pneumonia = Path(INPUT_PATH + '/train/PNEUMONIA').glob('*.jpeg')



# ---------------------------------------------------------------

# Train data format in (img_path, label) 

# Labels for [ the normal cases = 0 ] & [the pneumonia cases = 1]

# ---------------------------------------------------------------

normal_data = [(image, 0) for image in train_normal]

pneumonia_data = [(image, 1) for image in train_pneumonia]



train_data = normal_data + pneumonia_data



# Get a pandas dataframe from the data we have in our list 

train_data = pd.DataFrame(train_data, columns=['image', 'label'])



# Checking the dataframe...

train_data.head()
# Checking the dataframe...

train_data.tail()
# Shuffle the data 

train_data = train_data.sample(frac=1., random_state=100).reset_index(drop=True)



# Checking the dataframe...

train_data.head(10)
print(train_data)
# Counts for both classes

count_result = train_data['label'].value_counts()

print('Total of Train Data : ', len(train_data), '  (0 : Normal; 1 : Pneumonia)')

print(count_result)



# Plot the results 

plt.figure(figsize=(8,5))

sns.countplot(x = 'label', data =  train_data)

plt.title('Number of classes', fontsize=16)

plt.xlabel('Class type', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.xticks(range(len(count_result.index)), 

           ['Normal : 0', 'Pneumonia : 1'], 

           fontsize=14)

plt.show()
fig, ax = plt.subplots(3, 4, figsize=(20,15))

for i, axi in enumerate(ax.flat):

    image = imread(train_data.image[i])

    axi.imshow(image, cmap='bone')

    axi.set_title(('Normal' if train_data.label[i] == 0 else 'Pneumonia') 

                  + '  [size=' + str(image.shape) +']',

                  fontsize=14)

    axi.set(xticks=[], yticks=[])
train_data.to_numpy().shape
# ----------------------------------------------------------------------

#  Loading X-ray Images datasets from file 3 directories, respectively. 

# ----------------------------------------------------------------------

def load_data(files_dir='/train'):

    # list of the paths of all the image files

    normal = Path(INPUT_PATH + files_dir + '/NORMAL').glob('*.jpeg')

    pneumonia = Path(INPUT_PATH + files_dir + '/PNEUMONIA').glob('*.jpeg')



    # --------------------------------------------------------------

    # Data-paths' format in (img_path, label) 

    # labels : for [ Normal cases = 0 ] & [ Pneumonia cases = 1 ]

    # --------------------------------------------------------------

    normal_data = [(image, 0) for image in normal]

    pneumonia_data = [(image, 1) for image in pneumonia]



    image_data = normal_data + pneumonia_data



    # Get a pandas dataframe for the data paths 

    image_data = pd.DataFrame(image_data, columns=['image', 'label'])

    

    # Shuffle the data 

    image_data = image_data.sample(frac=1., random_state=100).reset_index(drop=True)

    

    # Importing both image & label datasets...

    x_images, y_labels = ([data_input(image_data.iloc[i][:]) for i in range(len(image_data))], 

                         [image_data.iloc[i][1] for i in range(len(image_data))])



    # Convert the list into numpy arrays

    x_images = np.array(x_images)

    y_labels = np.array(y_labels)

    

    print("Total number of images: ", x_images.shape)

    print("Total number of labels: ", y_labels.shape)

    

    return x_images, y_labels
# ---------------------------------------------------------

#  1. Resizing all the images to 224x224 with 3 channels.

#  2. Then, normalize the pixel values.  

# ---------------------------------------------------------

def data_input(dataset):

    # print(dataset.shape)

    for image_file in dataset:

        image = cv2.imread(str(image_file))

        image = cv2.resize(image, (224,224))

        if image.shape[2] == 1:

            # np.dstack(): Stack arrays in sequence depth-wise 

            #              (along third axis).

            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dstack.html

            image = np.dstack([image, image, image])

        

        # ----------------------------------------------------------

        # cv2.cvtColor(): The function converts an input image 

        #                 from one color space to another. 

        # [Ref.1]: "cvtColor - OpenCV Documentation"

        #     - https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html

        # [Ref.2]: "Python计算机视觉编程- 第十章 OpenCV" 

        #     - https://yongyuan.name/pcvwithpython/chapter10.html

        # ----------------------------------------------------------

        x_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        

        # Normalization

        x_image = x_image.astype(np.float32)/255.

        return x_image
# Import train dataset...

x_train, y_train = load_data(files_dir='/train')



print(x_train.shape)

print(y_train.shape)
x_train[0].shape
x_train[0]
y_train
# Import validation dataset...

x_val, y_val = load_data(files_dir='/val')



print(x_val.shape)

print(y_val.shape)
y_val
# Import test dataset...

x_test, y_test = load_data(files_dir='/test')



print(x_test.shape)

print(y_test.shape)
# Counts for both classes

count_result = pd.Series(y_test).value_counts()

print('Total of Test Data : ', len(y_test), '  (0 : Normal; 1 : Pneumonia)')

print('------------------')

print(count_result)

print('------------------')

print('1 :  ', count_result[1]/sum(count_result))

print('0 :  ', count_result[0]/sum(count_result))
y_test[:10]
model = Sequential([

    Conv2D(32, (5,5), activation='relu', padding='same', 

           input_shape=(224,224,3), name='Conv1_1'),

    BatchNormalization(name='bn1_1'),

    Conv2D(32, (5,5), activation='relu', padding='same', name='Conv1_2'),

    BatchNormalization(name='bn1_2'),

    Conv2D(32, (5,5), activation='relu', padding='same', name='Conv1_3'),

    BatchNormalization(name='bn1_3'),

    MaxPooling2D((2,2), name='MaxPool1'),

    Dropout(0.25),

    

    Conv2D(48, (3,3), activation='relu', padding='same', name='Conv2_1'),

    BatchNormalization(name='bn2_1'),

    Conv2D(48, (3,3), activation='relu', padding='same', name='Conv2_2'),

    BatchNormalization(name='bn2_2'),

    Conv2D(48, (3,3), activation='relu', padding='same', name='Conv2_3'),

    BatchNormalization(name='bn2_3'),    

    MaxPooling2D((2,2), name='MaxPool2'),

    Dropout(0.25),



    Conv2D(64, (3,3), activation='relu', padding='same', name='Conv3_1'),

    BatchNormalization(name='bn3_1'),

    Conv2D(64, (3,3), activation='relu', padding='same', name='Conv3_2'),

    BatchNormalization(name='bn3_2'),

    Conv2D(64, (3,3), activation='relu', padding='same', name='Conv3_3'),

    BatchNormalization(name='bn3_3'),

    MaxPooling2D((2,2), name='MaxPool3'),

    Dropout(0.25),

    

    # ----------------------------------------------------------------------

    # Using "1x1 convolution layer" to lower the complexity of computing

    # [Ref]: Prof Andrew Ng, "Inception Module", 

    #        https://www.youtube.com/watch?v=KfV8CJh7hE0

    # ----------------------------------------------------------------------

    Conv2D(64, (1,1), activation='relu', padding='same', name='Conv4_1_1x1'),

    BatchNormalization(name='bn4_1_1x1'),

    Conv2D(128, (3,3), activation='relu', padding='same', name='Conv4_2'),

    BatchNormalization(name='bn4_2'),

    MaxPooling2D((2,2), name='MaxPool4'),

    Dropout(0.25),



    # Using "1x1 convolution layer" 

    Conv2D(128, (1,1), activation='relu', padding='same', name='Conv5_1_1x1'),

    BatchNormalization(name='bn5_1_1x1'),

    Conv2D(256, (3,3), activation='relu', padding='same', name='Conv5_2'),

    BatchNormalization(name='bn5_2'),

    MaxPooling2D((2,2), name='MaxPool5'),

    Dropout(0.25),

    

    # Using "1x1 convolution layer" 

    Conv2D(256, (1,1), activation='relu', padding='same', name='Conv6_1x1'),

    BatchNormalization(name='bn6_1x1'),

    Conv2D(512, (3,3), activation='relu', name='Conv6_2'),

    BatchNormalization(name='bn6_2'),

    Dropout(0.5),

    

    Flatten(),

    Dense(64, activation='relu', name='fc'), 

    BatchNormalization(name='bn_fc'),

    Dropout(0.25),

    Dense(1, activation='sigmoid', name='Output') 

])
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True, dpi=85)
batch_size = 16

epochs_stage_1 = 10

epochs_stage_2 = 20

train_data_num = 4200
# Adam Optimizer with Learning-rate Decay 

basic_learning_rate = 0.001

opt = Adam(lr=basic_learning_rate, decay=basic_learning_rate/10.)



model.compile(optimizer=opt,

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])
## data_augmentation = False

print('Not using data augmentation.')

epochs = epochs_stage_1

history_no_data_aug = model.fit(x_train[:train_data_num], y_train[:train_data_num],

                               batch_size=batch_size,

                               epochs=epochs,

                               validation_data=(x_train[train_data_num:], y_train[train_data_num:]),

                               # validation_data=(x_val, y_val),

                               shuffle=False)
history_no_data_aug.history.keys()
acc = history_no_data_aug.history['accuracy']

val_acc = history_no_data_aug.history['val_accuracy']



loss = history_no_data_aug.history['loss']

val_loss = history_no_data_aug.history['val_loss']



epochs_range = range(1, epochs + 1)



plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylim(0, 1)

plt.xticks(epochs_range)

plt.title('Training and Validation Accuracy - without Data Augmentation')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylim(0, 1)

plt.xticks(epochs_range)

plt.title('Training and Validation Loss - without Data Augmentation')

plt.show()
# Score trained model.

loss, acc = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', loss)

print('Test accuracy:', acc)
# Get predictions

preds = model.predict(x_test)
preds.shape
y_pred = []

for i in range(len(preds)):

    if preds[i] > 0.5 : 

        y_pred.append(1) 

    else: 

        y_pred.append(0)

        

print(' y_pred = ', np.array(y_pred[:10]))

print(' y_test = ', y_test[:10])
mat = confusion_matrix(y_test, y_pred)

print(mat)



plt.figure(figsize=(8,6))

sns.heatmap(mat, square=False, annot=True, fmt ='d', cbar=True, annot_kws={"size": 16})

plt.title('0 : Normal   1 : Pneumonia', fontsize = 20)

plt.xticks(fontsize = 16)

plt.yticks(fontsize = 16)

plt.xlabel('predicted value', fontsize = 20)

plt.ylabel('true value', fontsize = 20)

plt.show()
# Calculate Precision and Recall

tn, fp, fn, tp = mat.ravel()

print('tn = {}, fp = {}, fn = {}, tp = {} '.format(tn, fp, fn, tp))



precision = tp/(tp+fp)

recall = tp/(tp+fn)

accuracy = (tp+tn)/(tp+tn+fp+fn)

f1_score = 2. * precision * recall / (precision + recall)

f2_score = 5. * precision * recall / (4. * precision + recall)



print("\nTest Recall of the model \t = {:.4f}".format(recall))

print("Test Precision of the model \t = {:.4f}".format(precision))

print("Test Accuracy of the model \t = {:.4f}".format(accuracy))

print("\nTest F1 score of the model \t = {:.4f}".format(f1_score))

print("\nTest F2 score of the model \t = {:.4f}".format(f2_score))
# Adam Optimizer with Learning-rate Decay 

lr_with_decay = basic_learning_rate / 10.

opt = Adam(lr=lr_with_decay, decay=lr_with_decay/100.)



model.compile(optimizer=opt,

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])
def data_augm():

    print('Using real-time data augmentation.')

    # This will do preprocessing and realtime data augmentation:

    datagen = ImageDataGenerator(

        # randomly shift images horizontally (fraction of total width)

        width_shift_range=0.05,

        # randomly shift images vertically (fraction of total height)

        height_shift_range=0.05,

        # rotation_range=20,

        horizontal_flip=True,  # Randomly flip inputs horizontally.

        # vertical_flip=True,  # Randomly flip inputs vertically.

        # zoom_range=[0.95, 1.05] # Range for random zoom

    )

    return datagen
print('With data augmentation.')

datagen = data_augm()

epochs = epochs_stage_2



# Compute quantities required for feature-wise normalization

# (std, mean, and principal components if ZCA whitening is applied).

datagen.fit(x_train[:train_data_num])



# Fit the model on the batches generated by datagen.flow().

history_data_aug = model.fit_generator(datagen.flow(x_train[:train_data_num], y_train[:train_data_num], 

                                                    batch_size=batch_size),

                                                    epochs=epochs,

                                                    validation_data=(x_train[train_data_num:], y_train[train_data_num:]),

                                                    # validation_data=(x_val, y_val),

                                                    workers=4)
acc = history_data_aug.history['accuracy']

val_acc = history_data_aug.history['val_accuracy']



loss = history_data_aug.history['loss']

val_loss = history_data_aug.history['val_loss']



epochs_range = range(1, epochs + 1)



plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylim(0, 1)

plt.xticks(epochs_range)

plt.title('Training and Validation Accuracy with Data Augmentation')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylim(0, 1)

plt.xticks(epochs_range)

plt.title('Training and Validation Loss with Data Augmentation')

plt.show()
# Score trained model.

loss, acc = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', loss)

print('Test accuracy:', acc)
# Get predictions

preds = model.predict(x_test)
preds.shape
y_pred = []

for i in range(len(preds)):

    if preds[i] > 0.5 : 

        y_pred.append(1) 

    else: 

        y_pred.append(0)

        

print(' y_pred = ', np.array(y_pred[:10]))

print(' y_test = ', y_test[:10])
mat = confusion_matrix(y_test, y_pred)

print(mat)



plt.figure(figsize=(8,6))

sns.heatmap(mat, square=False, annot=True, fmt ='d', cbar=True, annot_kws={"size": 16})

plt.title('0 : Normal   1 : Pneumonia', fontsize = 20)

plt.xticks(fontsize = 16)

plt.yticks(fontsize = 16)

plt.xlabel('predicted value', fontsize = 20)

plt.ylabel('true value', fontsize = 20)

plt.show()
# Calculate Precision and Recall

tn, fp, fn, tp = mat.ravel()

print('tn = {}, fp = {}, fn = {}, tp = {} '.format(tn, fp, fn, tp))



precision = tp/(tp+fp)

recall = tp/(tp+fn)

accuracy = (tp+tn)/(tp+tn+fp+fn)

f1_score = 2. * precision * recall / (precision + recall)

f2_score = 5. * precision * recall / (4. * precision + recall)



print("\nTest Recall of the model \t = {:.4f}".format(recall))

print("Test Precision of the model \t = {:.4f}".format(precision))

print("Test Accuracy of the model \t = {:.4f}".format(accuracy))

print("\nTest F1 score of the model \t = {:.4f}".format(f1_score))

print("\nTest F2 score of the model \t = {:.4f}".format(f2_score))
acc_total = history_no_data_aug.history['accuracy'] + history_data_aug.history['accuracy']

val_acc_total = history_no_data_aug.history['val_accuracy'] + history_data_aug.history['val_accuracy']



loss_total = history_no_data_aug.history['loss'] + history_data_aug.history['loss']

val_loss_total = history_no_data_aug.history['val_loss'] + history_data_aug.history['val_loss']
initial_epochs = epochs_stage_1

total_epochs = epochs_stage_1 + epochs_stage_2

epochs_range = range(1, total_epochs + 1)



plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)

plt.plot(epochs_range, acc_total, label='Training Accuracy')

plt.plot(epochs_range, val_acc_total, label='Validation Accuracy')

plt.ylim([0, 1])

plt.xticks(range(1,total_epochs+1,1))

plt.plot([initial_epochs,initial_epochs],

          plt.ylim(), label='Start Fine Tuning')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(epochs_range, loss_total, label='Training Loss')

plt.plot(epochs_range, val_loss_total, label='Validation Loss')

plt.ylim([0, 1])

plt.xticks(range(1,total_epochs+1,1))

plt.plot([initial_epochs,initial_epochs],

         plt.ylim(), label='Start Fine Tuning')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.xlabel('epoch')

plt.show()
# Saving the entire model to a HDF5 file：

# The '.h5' extension is for the HDF5 format.

model.save('PD_HDF5_model.h5')
# Reloading the HDF5 model, including its weights and the optimizer.

HDF5_model = tf.keras.models.load_model('PD_HDF5_model.h5')



# Show the model architecture

HDF5_model.summary()
# Evaluate the restored HDF5 model

loss, acc = HDF5_model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', loss)

print('Test accuracy:', acc)
# submission = pd.concat([pd.Series(range(1,(len(pred)+1)),name = "ImageId"),preds],axis = 1)

data_subm = {'ImageId': pd.Series(range(1,(len(y_pred)+1))), 'Prediction': y_pred}

submission = pd.DataFrame(data_subm)

submission = submission.applymap(str)



submission.to_csv("submission.csv",index=False)