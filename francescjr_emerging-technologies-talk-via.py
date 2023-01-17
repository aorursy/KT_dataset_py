import warnings

warnings.filterwarnings("ignore")

import numpy as np # linear algebra

import keras

from keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the "../input/" directory.

import os



# Specify directory to data for training

data_path = '/kaggle/input/lego-vs-unknown-cropped/VIA_dataset_cropped/train'

unknown_training_data = os.listdir(os.path.join(data_path, 'Unknown'))

lego_training_data = os.listdir(os.path.join(data_path, 'LEGO'))



unknown_samples = len(unknown_training_data)

lego_samples = len(lego_training_data)

print('There are %d unknown bricks and %d LEGO bricks' % (unknown_samples, lego_samples))
# First, lets inspect briefly the training data

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12,15))



for idx, i in enumerate([10, 1000, 5000, 6000, 7000]):

    img = plt.imread(os.path.join(data_path, 'Unknown', unknown_training_data[i]))

    axes[idx][0].imshow(img)

    axes[idx][0].set_title(unknown_training_data[i])

    axes[idx][0].set_axis_off()



for idx, i in enumerate([10, 1000, 5000, 8000, 6000]):

    img = plt.imread(os.path.join(data_path, 'LEGO', lego_training_data[i]))

    axes[idx][1].imshow(img)

    axes[idx][1].set_title(lego_training_data[i])

    axes[idx][1].set_axis_off()



plt.tight_layout()
# Parameters that can be modified and will alter network results!



# Percentage of training data used for validation

val_split = 0.2

train_samples = (lego_samples + unknown_samples) * (1 - val_split)

val_samples = (lego_samples + unknown_samples) * val_split

# Batch size for training (number of images fed at each step of an epoch)

bs = 128

# Number of epochs to train the network for

ep = 250

# Input image size (if using pretrained ResNet, it cannot be changed)

input_size = (224, 224)



# Keras Callbacks: set of functions that can be passed to the training procedure

# so that these are triggered while the model is training.

from keras.callbacks import EarlyStopping

# Early Stopping: We can set up a network to train for millions of epoch (why not). 

# However, at some point in time our network will stop learning, since no more can be

# learnt for the given problem. Early Stopping will detect that the model is not improving

# and it will stop the training process after a certain number of epoch.

pat = 10 # give the model 25 epoch to try and reduce val_loss again before stopping

es = EarlyStopping(monitor='val_loss', mode='min', patience=pat, restore_best_weights=True)
# Prepare the data for training!

# ImageDataGenerator is a Generator provided by Keras that will load the required data at each

# training batch, and perform the specified operations on it. In this example, we are normalizing

# the pixel values to the range [0, 1], resizing the images to [224, 224] pixels in width*height,

# and saving val_split percentage of the dataset for validation during train time

data_generator = ImageDataGenerator(rescale=1./255, vertical_flip=True, horizontal_flip=True,

                                   validation_split=val_split)

train_generator = data_generator.flow_from_directory(data_path, target_size=input_size, 

                                                    color_mode='rgb', batch_size=bs,

                                                    class_mode='categorical', seed=42,

                                                    subset='training')

validation_generator = data_generator.flow_from_directory(data_path, target_size=input_size, 

                                                    color_mode='rgb', batch_size=bs,

                                                    class_mode='categorical', seed=42,

                                                    subset='validation')

print('Categorical indexes for each class: ', train_generator.class_indices)

assert(train_generator.class_indices, validation_generator.class_indices)

lego_cat = train_generator.class_indices['LEGO']

unknown_cat = train_generator.class_indices['Unknown']

# This will load the images as needed for training, with the given input size, and the labels

# set as arrays of [0,1] or [1,0]
# Get a state of the art model, and just add a new layer at the end for our task

# This is known as TRANSFER LEARNING

# Basically, we get a well trained model (usually around weeks of training time), and swap its last

# classification layer for a new one, which we will train with our own data. 

# from keras.applications.resnet50 import ResNet50

# from keras.layers import Dense

# from keras.models import Model



# Load model ResNet50 from Keras database, exclude classification layers

# base_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')



# Append our new top layers. Last layer MUST contain same number of neurons as num_classes

# x = base_model.output

# x = Dense(1024, activation='relu')(x)

# x = Dense(512, activation='relu')(x)

# predictions = Dense(2, activation='softmax')(x)



# model = Model(inputs=base_model.input, outputs=predictions)



# We do not want to train the whole network, as this would take too much time

# First, we "freeze" all but our layers, and will only train these last ones

# This is always a good idea to verify that our approach converges to a solution

# for layer in base_model.layers:

#     layer.trainable = False
# Let's create our first Deep Learning model

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Activation

from keras.layers import BatchNormalization, GlobalAveragePooling2D

from keras.models import Model



entry = Input(shape=input_size + (3,))

x = Conv2D(64, 7, strides=2, padding='same')(entry)

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = MaxPooling2D()(x)

# outputs -> (112,112,64)



# 1st layer

x = Conv2D(64, 1, padding='same')(x)

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = Conv2D(64, 3, padding='same')(x)

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = MaxPooling2D()(x)

# outputs -> (56,56,64)



# 2nd layer

x = Conv2D(128, 1, padding='same')(x)

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = Conv2D(128, 3, padding='same')(x)

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = MaxPooling2D()(x)

# outputs -> (28,28,128)



# 3rd layer

x = Conv2D(256, 1, padding='same')(x)

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = Conv2D(256, 3, padding='same')(x)

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = MaxPooling2D()(x)

# outputs -> (14,14,256)



# 4th layer

# x = Conv2D(512, 1, padding='same')(x)

# x = BatchNormalization()(x)

# x = Activation('relu')(x)

# x = Conv2D(512, 3, padding='same')(x)

# x = BatchNormalization()(x)

# x = Activation('relu')(x)

# x = MaxPooling2D()(x)

# # outputs -> (7,7,512)



# # 5th layer

# x = Conv2D(1024, 1, padding='same')(x)

# x = BatchNormalization()(x)

# x = Activation('relu')(x)

# x = Conv2D(1024, 3, padding='same')(x)

# x = BatchNormalization()(x)

# x = Activation('relu')(x)

x = GlobalAveragePooling2D()(x)

# outputs -> (1,1024)



# Fully-connected layers for classification

x = Dense(256, activation='relu')(x)

x = Dense(64, activation='relu')(x)

predictions = Dense(2, activation='softmax')(x)



model = Model(inputs=entry, outputs=predictions)
# Specify the compiler and loss algorithm to train our network:

# - Nadam is considered the best optimzer for classification tasks, although its math is complex, 

# give it a try. SGD (Stochastic Gradient Descent) is the starting point when reading papers from

# ML world, so give that one a try too. 

# - Categorical crossentropy will calculate the entropy between the different class categories. Since

# this is logarithmic, an incorrect classification will penalize more than randomly selecting values, 

# which fits our purpose perfectly here. Since we have 2 classes, this would be the same as using

# binary_crossentropy

model.compile(optimizer='nadam', loss='categorical_crossentropy',

             metrics=['accuracy'])



# Visualize what our model and parameters look like

print(model.summary())
# It's time to train the model with our Generator Data

history = model.fit_generator(generator=train_generator, steps_per_epoch=train_samples//bs,

                              epochs=ep, validation_data=validation_generator,

                              validation_steps=val_samples//bs, callbacks=[es])
# Training the network has finished, now we can visualize how it went by plotting the accuracy

# and loss graphs

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,10))

ax[0].plot(history.history['accuracy'])

ax[0].plot(history.history['val_accuracy'])

ax[0].set_title('Model accuracy')

ax[0].set_xlabel('epoch')

ax[0].set_ylabel('accuracy')

ax[0].legend(['train', 'validation'])



ax[1].plot(history.history['loss'])

ax[1].plot(history.history['val_loss'])

ax[1].set_title('Model loss')

ax[1].set_xlabel('epoch')

ax[1].set_ylabel('loss')

ax[1].legend(['train', 'validation'])
test_data_path = '/kaggle/input/lego-vs-unknown-cropped/VIA_dataset_cropped/test'

unknown_test_data = os.listdir(os.path.join(test_data_path, 'Unknown'))

lego_test_data = os.listdir(os.path.join(test_data_path, 'LEGO'))



unknown_test_samples = len(unknown_test_data)

lego_test_samples = len(lego_test_data)

print('There are %d unknown bricks and %d LEGO bricks' % (unknown_test_samples, lego_test_samples))
# We can verify that the images are similar in the test set and training set

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12,15))



for idx, i in enumerate([100, 200, 300, 400, 500]):

    img = plt.imread(os.path.join(test_data_path, 'Unknown', unknown_test_data[i]))

    axes[idx][0].imshow(img)

    axes[idx][0].set_title(unknown_test_data[i])

    axes[idx][0].set_axis_off()



for idx, i in enumerate([100, 200, 300, 400, 500]):

    img = plt.imread(os.path.join(test_data_path, 'LEGO', lego_test_data[i]))

    axes[idx][1].imshow(img)

    axes[idx][1].set_title(lego_test_data[i])

    axes[idx][1].set_axis_off()



plt.tight_layout()
from keras.preprocessing.image import load_img, img_to_array

# First, we create our array of true predictions, which contains 0s and 1s for each

# entry of each class

y_true = np.concatenate((np.zeros(lego_test_samples),

                         np.ones(unknown_test_samples)))

# We will load images 1 by 1, class by class (instead of using the generator)

# starting with LEGO test bricks

lego_class_results = []

for image_name in lego_test_data:

    # load image, convert to array and add the batch layer (axis=0)

    img = load_img(os.path.join(test_data_path, 'LEGO', image_name),

                  target_size=input_size, grayscale=False)

    img = img_to_array(img)

    # apply pixel value normalization

    img = img * 1./255

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[:, -1] # is an array of the probability of each class,

    # which adds up to 1.0 in total.

    lego_class_results.append(prediction)

lego_class_results = np.array(lego_class_results)

# And repeat again for the other class

unknown_class_results = []

for image_name in unknown_test_data:

    img = load_img(os.path.join(test_data_path, 'Unknown', image_name),

                  target_size=input_size)

    img = img_to_array(img)

    # apply pixel value normalization

    img = img * 1./255

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[:, -1] 

    unknown_class_results.append(prediction)

unknown_class_results = np.array(unknown_class_results)

y_pred = np.concatenate((lego_class_results, unknown_class_results))
# We can check some random predictions on test set

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12,15))



cls_idx = {0: 'LEGO', 1: 'Unknown'}



for idx, i in enumerate([100, 200, 300, 400, 500]):

    img = plt.imread(os.path.join(test_data_path, 'Unknown', unknown_test_data[i]))

    axes[idx][0].imshow(img)

    pred = 1 if unknown_class_results[i] > 0.5 else 0

    axes[idx][0].set_title(f'True label: Unknown, Predicted: {cls_idx[pred]}')

    axes[idx][0].set_axis_off()



for idx, i in enumerate([100, 200, 300, 400, 500]):

    img = plt.imread(os.path.join(test_data_path, 'LEGO', lego_test_data[i]))

    axes[idx][1].imshow(img)

    pred = 1 if lego_class_results[i] > 0.5 else 0

    axes[idx][1].set_title(f'True label: LEGO, Predicted: {cls_idx[pred]}')

    axes[idx][1].set_axis_off()



plt.tight_layout()
# We have an array of probabilities for each input image.

# We want to calculate, at different THRESHOLD levels, what the performance of our method is

tpr_list = []

fpr_list = []

prec_list = []

rec_list = []

f1_list = []

thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for th in thresholds:

    fp = sum(lego_class_results > th)

    tn = np.abs(fp - lego_test_samples)

    tp = sum(unknown_class_results > th)

    fn = np.abs(tp - unknown_test_samples)

    print('At confidence level %.2f%%, the Confusion Matrix is as follows:' % (th*100))

    print('                              True Label ')

    print('                            LEGO   ||   Unknown ')

    print(' Predicted ||   LEGO    ||  %3d    ||    %3d ' % (tn, fp))

    print('   Label   ||  Unknown  ||  %3d    ||    %3d ' % (fn, tp))

    print('---------------------------------------------------------------')

    tpr = tp / (tp + fn)

    fpr = fp / (fp + tn)

    tpr_list.append(tpr)

    fpr_list.append(fpr)

    precision = tp  / (tp + fp + 1e-8)

    recall = tp / (tp + fn + 1e-8)

    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    prec_list.append(precision)

    rec_list.append(recall)

    f1_list.append(f1_score)
# Let's see the performance metrics of our model in a general table

print('| Threshold | Precision | Recall | F1 Score |   TPR  |   FPR  |')

print('|-------------------------------------------------------------|')

for th, p, r, f1, tpr, fpr in zip(thresholds, prec_list, rec_list, f1_list, tpr_list, fpr_list):

    print('|    %.2f   |   %.4f  | %.4f |  %.4f  | %.4f | %.4f |' % (th, p, r, f1, tpr, fpr))
# Finally, we can now generate the ROC Curve for our test dataset

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))

#ax.plot(fpr_list, tpr_list, 'ro-', label='Sample model ROC')

#ax.plot(list(np.linspace(0, 1, num = 10)), list(np.linspace(0, 1, num = 10)), 'bo--', label = 'Random Guess');



# the value at threshold=1 is of indeterminate form x/x = 0/0 = 1

rec_list[-1] = 0.0

prec_list[-1] = 1.0



ax.plot(rec_list, prec_list, 'ro-', label='Trained ML model')

class_sample_ratio = unknown_test_samples / (unknown_test_samples + lego_test_samples)

ax.plot([0, 1], [class_sample_ratio, class_sample_ratio], 'bo-', label='Random Guess')



for x,y,s in zip(rec_list, prec_list, thresholds):

    ax.text(x+0.004, y-0.02, str(s), fontdict={'size': 14})



ax.legend()

ax.set_ylabel('Precision')

ax.set_xlabel('Recall')

ax.set_title('Precision vs Recall Curve')
