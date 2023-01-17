# !pip install keras-vis
# Display GPU information of Kaggle Kernel

!nvidia-smi
from glob import glob

import os

import numpy as np

import pandas as pd

import random

from skimage.io import imread



import matplotlib.pyplot as plt

%matplotlib inline



from tensorflow.python.keras.models import load_model

from tensorflow.python.keras.applications import ResNet50

from tensorflow.python.keras.applications.resnet50 import preprocess_input, decode_predictions

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from tensorflow.python.keras.models import Model

# from keras.models import load_model



# from keras.applications import ResNet50

# from keras.applications.resnet50 import preprocess_input, decode_predictions

# from keras.models import Sequential

# from keras.layers import Dense, Flatten, GlobalAveragePooling2D

# from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img



# from keras.models import Model
# Path to data

data_dir  = '../input/chest-xray-pneumonia/chest_xray/chest_xray/'

train_dir = data_dir+'train/'

test_dir  = data_dir+'test/'

val_dir   = data_dir + 'val/'



# Get the path to the normal and pneumonia sub-directories

normal_cases_dir = train_dir + 'NORMAL/'

pneumonia_cases_dir = train_dir + 'PNEUMONIA/'



print("Datasets:",os.listdir(data_dir))

print("Train:\t", os.listdir(train_dir))

print("Test:\t", os.listdir(test_dir))
# Get the list of all the images

normal_cases = glob(normal_cases_dir+'/*.jpeg')

pneumonia_cases = glob(pneumonia_cases_dir+'/*.jpeg')



# An empty list. We will insert the data into this list in (img_path, label) format

train_data = []



# Go through all the normal cases. The label for these cases will be 0

for img in normal_cases:

    train_data.append((img,0))



# Go through all the pneumonia cases. The label for these cases will be 1

for img in pneumonia_cases:

    train_data.append((img, 1))



# Get a pandas dataframe from the data we have in our list 

train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None)



# Shuffle the data 

train_data = train_data.sample(frac=1.).reset_index(drop=True)



# Get few samples for both the classes

pneumonia_samples = (train_data[train_data['label']==1]['image'].iloc[:5]).tolist()

normal_samples = (train_data[train_data['label']==0]['image'].iloc[:5]).tolist()



# Concat the data in a single list and del the above two list

samples = pneumonia_samples + normal_samples

del pneumonia_samples, normal_samples



# Plot the data 

f, ax = plt.subplots(2,5, figsize=(30,10))

for i in range(10):

    img = imread(samples[i])

    ax[i//5, i%5].imshow(img, cmap='gray')

    if i<5:

        ax[i//5, i%5].set_title("Pneumonia")

    else:

        ax[i//5, i%5].set_title("Normal")

    ax[i//5, i%5].axis('off')

    ax[i//5, i%5].set_aspect('auto')

plt.show()
img = imread(samples[0])
img.shape
plt.imshow(img)
image_size = 150

nb_train_samples = 5216 # number of files in training set

batch_size = 16





## Specify the values for all arguments to data_generator_with_aug.

data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,

                                             horizontal_flip = True,

                                             width_shift_range = 0.2,

                                             height_shift_range = 0.2,

                                             shear_range = 0.2,

                                             zoom_range = 0.2)

            

data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)



train_generator = data_generator_with_aug.flow_from_directory(

       directory = '../input/chest-xray-pneumonia/chest_xray/chest_xray/train/',

       target_size = (image_size, image_size),

       batch_size = batch_size,

       class_mode = 'categorical')



validation_generator = data_generator_no_aug.flow_from_directory(

       directory = '../input/chest-xray-pneumonia/chest_xray/chest_xray/val/',

       target_size = (image_size, image_size),

       class_mode = 'categorical')



test_generator = data_generator_no_aug.flow_from_directory(

       directory = '../input/chest-xray-pneumonia/chest_xray/chest_xray/test/',

       target_size = (image_size, image_size),

       batch_size = batch_size,

       class_mode = 'categorical')



train_generator.image_shape
# Plot Loss  

def plot_loss(model):

    plt.plot(model.history.history['loss'])

    plt.plot(model.history.history['val_loss'])

    plt.title('Model Loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Training set', 'Test set'], loc='upper left')

    plt.show()

# Plot Accuracy 

def plot_accuracy(model):

    plt.plot(model.history.history['acc'])

    plt.plot(model.history.history['val_acc'])

    plt.title('Model Accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Training set', 'Validation set'], loc='upper left')

    plt.show()
# Choose a random image and apply model for prediction

def choose_image_and_predict():

    normal_or_pneumonia = ['NORMAL', 'PNEUMONIA']

    folder_choice = (random.choice(normal_or_pneumonia))

    

    pneumonia_images = glob('../input/chest-xray-pneumonia/chest_xray/chest_xray/val/'+folder_choice+'/*')

    img_choice = (random.choice(pneumonia_images))



    img = load_img(img_choice, target_size=(150, 150))

    img = img_to_array(img)

    plt.imshow(img / 255.)

    x = preprocess_input(np.expand_dims(img.copy(), axis=0))

#     pred_class = model.predict_classes(x)

    pred = model.predict(x)

    pred_class = pred.argmax(axis=-1)

    print("Actual class:", folder_choice)

    if pred_class[0] == 0:

        print("Predicted class: Normal")

        print("Likelihood:", pred[0][0].round(4))

        if pred[0][0].round(4) < 0.8:

            print("WARNING, low confidence")

    else:

        print("Predicted class: Pneumonia")

        print('Likelihood:', pred[0][1].round(4))

        if pred[0][1].round(4) < 0.8:

            print("WARNING, low confidence")        

EPOCHS = 4

STEPS = nb_train_samples / batch_size

num_classes = 2
# Load the pre-trained model

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



# Stack the pre-trained model with a fully-connected layer

model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))



# Add fully-connect layer (aka Dense layer) with softmax activation

model.add(Dense(num_classes, activation='softmax'))



# Freeze all layers except last one

model.layers[0].trainable = False



# Define loss function, optimizer and metrics

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# Load the pre-trained model

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

resnet = ResNet50(pooling='avg', 

                  weights=resnet_weights_path,

                  input_shape=(150, 150, 3),

                  include_top = False)



# Freeze all layers in Resnet

for layer in resnet.layers:

    layer.trainable = False



# Stack the pre-trained model with a fully-connected layer

# Add fully-connect layer (aka Dense layer) with softmax activation

x = resnet.output

x = Flatten()(x)

x = Dense(num_classes, activation='softmax')(x)



# Define loss function, optimizer and metrics

model = Model(inputs=resnet.input, outputs= x)

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit_generator(

       train_generator, # specify where model gets training data

       epochs = EPOCHS,

       steps_per_epoch=STEPS,

       validation_data=validation_generator) # specify where model gets validation data



# model.fit_generator(

#        train_generator, # specify where model gets training data

#        epochs = EPOCHS,

#        steps_per_epoch=STEPS,

#             validation_steps=4,

#        validation_data=validation_generator) # specify where model gets validation data



# Evaluate the model

scores = model.evaluate_generator(test_generator, steps=len(test_generator))

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# Evaluate the model

scores = model.evaluate_generator(test_generator,steps=len(test_generator))

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# Make a prediction

choose_image_and_predict()
plot_loss(model)
plot_accuracy(model)
# Save model

model.save('xray_model_resnet.h5')



# Load model

# model = load_model('xray_model.h5')