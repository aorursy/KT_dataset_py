import warnings
warnings.filterwarnings("ignore")
# import packages

import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt

import tensorflow.keras as keras
from keras import regularizers
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
from keras.preprocessing.image import ImageDataGenerator
# directory of input data
print(os.listdir("../input"))
# grab the train and test data into dataframes
df_train = pd.read_csv('../input/dog-breed-identification/labels.csv') 
df_test = pd.read_csv('../input/dog-breed-identification/sample_submission.csv') 
print('labels : train dataframe')
df_train.head()
print('samples : test dataframe')
df_test.head()
# the breed is what we are after for each dog sample
labels = df_train['breed']
# but we need to get dummy variables for the breed columns
one_hot = pd.get_dummies(labels, sparse = True)
print('get dummies for breed data')
one_hot
print('one hot encode the labels into numpy array')
one_hot_labels = np.asarray(one_hot)
one_hot_labels
# the images will be resized to 64x64
im_resize = 64
print('sample the dog images after resize efforts')
#visualize a dogger
dogger1 = df_train['id'][0]
dogger2 = df_train['id'][1]
dogger3 = df_train['id'][2]
dogger4 = df_train['id'][3]
pupper1 = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(dogger1))
pupper2 = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(dogger2))
pupper3 = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(dogger3), cv2.IMREAD_GRAYSCALE)
pupper4 = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(dogger4), cv2.IMREAD_GRAYSCALE)
pupper4 = cv2.resize(pupper4, (im_resize, im_resize))
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(pupper1)
axarr[0,1].imshow(pupper2)
axarr[1,0].imshow(pupper3,cmap="gray", vmin=0, vmax=255)
axarr[1,1].imshow(pupper4,cmap="gray", vmin=0, vmax=255)
plt.xticks([])
plt.yticks([])
im_size = pupper1.shape
print('Image original shape', im_size)
print('Image resized shape', cv2.resize(pupper1, (im_resize, im_resize)).shape)

x_train = []
y_train = []
x_test = []
print('arrays to hold x/y/train/test data initialized')
i = 0 
for f, breed in tqdm(df_train.values):
    img = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(f))
    img_resized = cv2.resize(img, (im_resize, im_resize))
    x_train.append(img_resized)
    label = one_hot_labels[i]
    y_train.append(label)
    i += 1
print('\nImages split into training sets, resized and one hot encoding labels appended.')
# no need for the big dataframe anymore
del df_train
print('df_train used; cleaned up.')
for f in tqdm(df_test['id'].values):
    img = cv2.imread('../input/dog-breed-identification/test/{}.jpg'.format(f))
    img_resized = cv2.resize(img, (im_resize, im_resize))
    x_test.append(img_resized)
print('\nTest data read and resized into x_test[].')
num_class = 120 #static ftw
print('Number of classes: ', num_class)
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, shuffle=True,  test_size=0.1)
print('Split into new values with train_test_split()')
del x_train, y_train
print('Initial values used; cleaned up.')
# author defines function to seek top 3 value
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
print('ImageDataGenerator being used to fit X_train.')

datagen = ImageDataGenerator(width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.2,
                            rotation_range=30,
                            vertical_flip=False,
                            horizontal_flip=True)


datagen.fit(X_train)
print('ResNet50 will load the `resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5` file included as pretrained model.\n')

base_model = ResNet50(weights="../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False, input_shape=(im_resize, im_resize, 3))

x = base_model.output
x = Flatten()(x) # Flattens the input. Does not affect the batch size.
x = Dense(512, activation='relu')(x)
x = Dropout(0.25)(x)
# Dropout consists in randomly setting
#     a fraction `rate` of input units to 0 at each update during training time,
#     which helps prevent overfitting.

logits = Dense(num_class, activation='softmax')(x)

model = Model(base_model.input, logits)

model.compile(optimizer='Adam',
          loss='categorical_crossentropy', 
           metrics=[categorical_crossentropy, categorical_accuracy])

print('\nThis base model has Flatten(), Dense(relu), Dropout() applied before a Dense(softmax)')
print('\nFinally coming to a Model(), where it is compiled with adam optimizer, loss of categorical_crossentropy, looking for accuracy as well.')
print('Graph generating helper function defined...')
def gen_graph(history, title):
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Accuracy ' + title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['categorical_crossentropy'])
    plt.plot(history.history['val_categorical_crossentropy'])
    plt.title('Loss ' + title)
    plt.ylabel('MLogLoss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
print('.flow() is used to train on the image data for X and Y arrays.')

train_generator = datagen.flow(np.array(X_train), np.array(Y_train), 
                               batch_size=32) 
# Takes data & label arrays, generates batches of augmented data.
print('.fit_generator() on the model using the train_generator just made and get the history.\n')

batch_size = 512

history_rmsprop = model.fit_generator(
    train_generator,
    epochs=30, steps_per_epoch=len(X_train) / batch_size,
    validation_data=(np.array(X_train), np.array(Y_train)), validation_steps=len(X_valid) / batch_size)

print('Use the model to run .predict() on x_test.')

preds = model.predict(np.array(x_test), verbose=1)
print('Accuracy plotted for predictions made.\n')

#plot the accuracy
gen_graph(history_rmsprop, 
              "ResNet50 RMSprop")
print('Dataframe is created with the prediction data and displayed.')

sub = pd.DataFrame(preds)
col_names = one_hot.columns.values
sub.columns = col_names
sub.insert(0, 'id', df_test['id'])
sub

print('The output dataframe created is saved to `output_rmsprop_aug.csv` and model saved as `rmsprop_v2_augmentation.h5`')

sub.to_csv('output_rmsprop_aug.csv', index=False)

model.save('rmsprop_v2_augmentation.h5')