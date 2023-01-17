import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.image import imread

# Technically not necessary in newest versions of jupyter

%matplotlib inline
import tensorflow as tf

import timeit



device_name = tf.test.gpu_device_name()

if "GPU" not in device_name:

    print("GPU device not found")

print('Found GPU at: {}'.format(device_name))
my_data_dir = '../input/cell-images-for-detecting-malaria/cell_images/cell_images'
# CONFIRM THAT THIS REPORTS BACK 'test', and 'train'

os.listdir(my_data_dir) 
train_path = my_data_dir



#test_path = my_data_dir+'/test/'

#train_path = my_data_dir+'/train/'
os.listdir(train_path)
os.listdir(train_path+'/Parasitized')[0:10]
os.listdir(train_path+'/Uninfected')
para_cell = train_path+'/Parasitized/'+os.listdir(train_path+'/Parasitized')[0]#
para_img= imread(para_cell)

para_img
plt.figure(figsize=(8,8))

plt.imshow(para_img)
para_img.shape
unifected_cell_path = train_path+'/Uninfected/'+os.listdir(train_path+'/Uninfected')[0]

unifected_cell = imread(unifected_cell_path)

plt.imshow(unifected_cell)
len(os.listdir(train_path+'/Parasitized'))
len(os.listdir(train_path+'/Uninfected'))
unifected_cell.shape
para_img.shape
# Other options: https://stackoverflow.com/questions/1507084/how-to-check-dimensions-of-all-images-in-a-directory-using-python

dim1 = []

dim2 = []

for image_filename in os.listdir(train_path+'/Uninfected'):

    

    if image_filename[-3:] == 'png':

        img = imread(train_path+'/Uninfected'+'/'+image_filename)

        d1,d2,colors = img.shape

        dim1.append(d1)

        dim2.append(d2)
sns.jointplot(dim1,dim2,color='red',kind='kde')

sns.jointplot(dim1,dim2,color='green',kind='hex')

sns.jointplot(dim1,dim2,color='grey',kind='reg')

help(sns.jointplot)
np.mean(dim1)
np.mean(dim2)
image_shape = (130,130,3)# we can play with this size to check if the performance improves
from tensorflow.keras.preprocessing.image import ImageDataGenerator
help(ImageDataGenerator)
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees

                               width_shift_range=0.10, # Shift the pic width by a max of 5%

                               height_shift_range=0.10, # Shift the pic height by a max of 5%

                               rescale=1/255, # Rescale the image by normalzing it.

                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)

                               zoom_range=0.1, # Zoom in by 10% max

                               horizontal_flip=True, # Allo horizontal flipping

                               fill_mode='nearest', # Fill in missing pixels with the nearest filled value

                                validation_split=0.1)#splits the training set in 90 train / 10 cvalidation
plt.imshow(para_img)
plt.imshow(image_gen.random_transform(para_img))
plt.imshow(image_gen.random_transform(para_img))
train_generator = image_gen.flow_from_directory(train_path,subset='training')
validation_generator = image_gen.flow_from_directory(train_path,subset='validation')
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
help(MaxPooling2D)
with tf.device('/GPU:0'):

    



    #https://stats.stackexchange.com/questions/148139/rules-for-selecting-convolutional-neural-network-hyperparameters

    model = Sequential()



    model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu',))

    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))



    model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))



    model.add(Conv2D(filters=128, kernel_size=(3,3),input_shape=image_shape, activation='relu',padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))





    model.add(Flatten())





    model.add(Dense(128))

    model.add(Activation('relu'))



    # Dropouts help reduce overfitting by randomly turning neurons off during training.

    # Here we say randomly turn off 50% of neurons.

    model.add(Dropout(0.5))



    # Last layer, remember its binary so we use sigmoid

    model.add(Dense(1))

    model.add(Activation('sigmoid'))



    model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=5)
help(image_gen.flow_from_directory)
batch_size = 32
train_image_gen = image_gen.flow_from_directory(train_path,

                                               target_size=image_shape[:2],

                                                color_mode='rgb',

                                               batch_size=batch_size,

                                               class_mode='binary',

                                               subset='training')
test_image_gen = image_gen.flow_from_directory(train_path,

                                               target_size=image_shape[:2],

                                               color_mode='rgb',

                                               batch_size=batch_size,

                                               class_mode='binary',shuffle=False,

                                                  subset='validation')
train_image_gen.class_indices
import warnings

warnings.filterwarnings('ignore')


# Configure the TensorBoard callback and fit your model



tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")

with tf.device('/GPU:0'):

    results = model.fit_generator(train_image_gen,epochs=10,

                              validation_data=test_image_gen,

                             callbacks=[early_stop,tensorboard_callback])
# Load the extension and start TensorBoard

%load_ext tensorboard



%tensorboard --logdir logs

%reload_ext tensorboard

%tensorboard --logdir logs

!kill 457
from tensorflow.keras.models import load_model
#model.save('malaria_detector.h5')
model = load_model('malaria_detector.h5')
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
losses[['accuracy','val_accuracy']].plot()
model.metrics_names
model.evaluate_generator(test_image_gen)
from tensorflow.keras.preprocessing import image
# https://datascience.stackexchange.com/questions/13894/how-to-get-predictions-with-predict-generator-on-streaming-test-data-in-keras

pred_probabilities = model.predict_generator(test_image_gen)
pred_probabilities
test_image_gen.classes
predictions = pred_probabilities > 0.5#this is the most important number for my

                                      #classification report
# Numpy can treat this as True/False for us

predictions
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(test_image_gen.classes,predictions))
#help(confusion_matrix)
confusion_matrix(test_image_gen.classes,predictions)
cm = confusion_matrix(test_image_gen.classes,predictions)
cm[0][1]
FP = []

FN = []

TP = []

TN = []



for i in range(1,100,1):

    

    predictions = pred_probabilities > (i/100)

    confusion_matrix(test_image_gen.classes,predictions)

    TP.append(confusion_matrix(test_image_gen.classes,predictions)[0][0])

    TN.append(confusion_matrix(test_image_gen.classes,predictions)[0][1])

    FP.append(confusion_matrix(test_image_gen.classes,predictions)[1][0])

    FN.append(confusion_matrix(test_image_gen.classes,predictions)[1][1])

    

#help(pd.DataFrame)
df_cm = pd.DataFrame(TP,columns=['TP'])



df_cm
df_cm['TN'] = TN

df_cm['FP'] = FP

df_cm['FN'] = FN
df_cm
plt.plot(range(1,100),df_cm[['TP','TN','FP','FN']])
df_cm.head()
sns.pairplot(data=df_cm)
plt.figure(figsize=(9,7))

sns.heatmap(df_cm.corr(),annot=True)
plt.plot(range(1,100),df_cm[['TP','N']])
# Your file path will be different!

para_cell
my_image = image.load_img(para_cell,target_size=image_shape)
my_image
type(my_image)
my_image = image.img_to_array(my_image)
type(my_image)
my_image.shape
my_image = np.expand_dims(my_image, axis=0)
my_image.shape
model.predict(my_image)
train_image_gen.class_indices
test_image_gen.class_indices