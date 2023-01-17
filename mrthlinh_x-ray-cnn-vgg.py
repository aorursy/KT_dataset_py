from glob import glob

import os

import numpy as np

import pandas as pd

import random

from skimage.io import imread

import cv2

import matplotlib.pyplot as plt

%matplotlib inline

# import h5py

from tensorflow.python.keras.models import load_model

from tensorflow.python.keras.applications import ResNet50

from tensorflow.python.keras.applications.resnet50 import preprocess_input, decode_predictions

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D,Dropout

from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from tensorflow.python.keras.callbacks import EarlyStopping

from tensorflow.python.keras.models import Model

import tensorflow.keras.backend as K

from sklearn.metrics import f1_score, accuracy_score
!ls ../input/chest-xray-pneumonia
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
train_pos = len(glob(train_dir+'PNEUMONIA/*.jpeg'))

train_neg = len(glob(train_dir+'NORMAL/*.jpeg'))

print("Traing set\t pos case: {}\t neg case: {}\t ratio: {}".format(train_pos,train_neg,train_pos / (train_neg + train_pos)))



val_pos = len(glob(val_dir+'PNEUMONIA/*.jpeg'))

val_neg = len(glob(val_dir+'NORMAL/*.jpeg'))

print("Validate set\t pos case: {}\t neg case: {}\t ratio: {}".format(val_pos,val_neg,val_pos / (val_neg + val_pos)))



test_pos = len(glob(test_dir+'PNEUMONIA/*.jpeg'))

test_neg = len(glob(test_dir+'NORMAL/*.jpeg'))

print("Test set\t pos case: {}\t neg case: {}\t ratio: {}".format(test_pos,test_neg,test_pos / (test_neg + test_pos)))

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
samples[0]
imread(samples[0]).shape
imread(samples[0])
image_size = 150

nb_train_samples = 5216 # number of files in training set

batch_size = 16

# batch_size = 32





## Specify the values for all arguments to data_generator_with_aug.

data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,

                                             horizontal_flip = True,

                                             width_shift_range = 0.2,

                                             height_shift_range = 0.2,

                                             shear_range = 0.2,

                                             zoom_range = 0.2)

            

data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)



train_generator = data_generator_with_aug.flow_from_directory(

       directory = train_dir,

       target_size = (image_size, image_size),

       batch_size = batch_size,

       class_mode = 'categorical')



validation_generator = data_generator_no_aug.flow_from_directory(

       directory = val_dir,

       target_size = (image_size, image_size),

       class_mode = 'categorical')



test_generator = data_generator_no_aug.flow_from_directory(

       directory = test_dir,

       target_size = (image_size, image_size),

       batch_size = batch_size,

       class_mode = 'categorical')
train_generator.next()[0][0,:]
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



    img = load_img(img_choice, target_size=(image_size, image_size))

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

def evaluation_result():

    # Evaluation the results

    y_prob = model.predict_generator(test_generator)

    y_pred = np.argmax(y_prob, axis = -1)

    y_true = test_generator.labels



    f1 = f1_score(y_pred=y_pred, y_true=y_true)

    print("F1 score: {}".format(f1))



    acc = accuracy_score(y_pred=y_pred, y_true=y_true)

    print("Accuracy: {}".format(acc))



    # True positive rate TPR = TP / P

    # True negative rate TNR = TN / N

    P = len(y_pred[y_pred == 1])

    N = len(y_pred[y_pred == 0])



    TP = len(y_pred[(y_pred == 1) & (y_true == 1)])

    TN = len(y_pred[(y_pred == 0) & (y_true == 0)])



    if P != 0: 

        TPR = 100 * TP / P

        print("True Positive Rate: {}".format(TPR))



        # False Negative Rate FN / P

        FN = len(y_pred[(y_pred == 0) & (y_true == 1)])

        FNR = 100 * FN / P

        print("False Negative Rate: {}".format(FNR))

    else:

        print("All sample tests are 0")



    if N != 0:

        TNR = 100 * TN / N

        print("True Negative Rate: {}".format(TNR))

    else:

        print("All sample tests are 1")

from tensorflow.python.keras.applications import VGG16

from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
num_classes = 2

EPOCHS = 20

STEPS = nb_train_samples / batch_size
# # Load the pre-trained model

# vgg_weights_path = '../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'



# model = Sequential()

# model.add(VGG16(include_top=False, pooling='avg', weights=vgg_weights_path))

# model.add(Dense(units=2, activation='softmax'))

# model.layers[0].trainable = False



# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
# model.fit_generator(

#     train_generator, # specify where model gets training data

#     epochs = EPOCHS,

#     steps_per_epoch=STEPS,

#     validation_data=validation_generator) # specify where model gets validation data



# # Evaluate the model

# scores = model.evaluate_generator(test_generator)

# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# evaluation_result()
# Load the pre-trained model

vgg_weights_path = '../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

vgg16 = VGG16(pooling='avg', 

                  weights=vgg_weights_path,

                  input_shape=(image_size, image_size, 3),

                  include_top = False)



# Freeze all layers in Resnet

for layer in vgg16.layers:

    layer.trainable = False



# Stack the pre-trained model with a fully-connected layer

# Add fully-connect layer (aka Dense layer) with softmax activation

x = vgg16.output

# x = Flatten()(x)

# x = Dense(num_classes, activation='softmax')(x)



# FC layer

x = Flatten()(x)

x = Dense(units=4096, activation='relu')(x)

# x = Dropout(rate=0.3)(x)

# x = Dense(units=4096, activation='relu')(x)

# x = Dropout(rate=0.3)(x)

x = Dense(units=2, activation='softmax')(x)







# Define loss function, optimizer and metrics

model = Model(inputs=vgg16.input, outputs= x)



optimizer = Adam(lr = 0.0001)

early_stopping_monitor = EarlyStopping(patience = 3, monitor = "val_acc", mode="max", verbose = 2)

model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit_generator(

    train_generator, # specify where model gets training data

    epochs = EPOCHS,

    steps_per_epoch=STEPS,

    validation_data=validation_generator, # specify where model gets validation data

    callbacks=[early_stopping_monitor])



# history = model.fit_generator(epochs=5, callbacks=[early_stopping_monitor], shuffle=True, validation_data=val_batches, generator=train_batches, steps_per_epoch=500, validation_steps=10,verbose=2)



# Evaluate the model

scores = model.evaluate_generator(test_generator)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
evaluation_result()
# Make a prediction

choose_image_and_predict()
plot_loss(model)

plot_accuracy(model)
# Save model

model.save('xray_model_vgg16.h5')
!ls '../input/chest-xray-pneumonia/chest_xray/chest_xray/test/'
# path = '../input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL/'

path = '../input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/'

pneumonia_cases = glob(path+'/*.jpeg')

img_path = pneumonia_cases[0]



img = load_img(img_path, target_size=(image_size,image_size))

x = img_to_array(img)

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)
pred = model.predict(x)

pred_class = pred.argmax(axis=-1)[0]

class_output = model.output[:, pred_class]

last_conv_layer = model.get_layer("block5_conv3")



grads = K.gradients(class_output, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):

    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)

heatmap /= np.max(heatmap)



img = cv2.imread(img_path)

# img = img_to_array(img)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
f, ax = plt.subplots(1,2, figsize=(30,10))

ax[0].imshow(img, cmap='gray')

ax[1].imshow(superimposed_img, cmap='gray')