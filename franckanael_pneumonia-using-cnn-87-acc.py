import os

import matplotlib.pyplot as plt

import numpy as np

import cv2 as cv

import seaborn as sns



import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization, Activation, Input, Concatenate, Layer

from keras.regularizers import l2

from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix
labels = ['NORMAL', 'PNEUMONIA']

BATCH_SIZE = 32

IMG_HEIGHT = 224

IMG_WIDTH = 224

IMG_DEPTH = 1

EPOCH = 12
def get_images(path_image):

  list_images = []

  for label in labels:

    path_image_label = path_image + '/' + label

    fichiers = [f for f in os.listdir(path_image_label) if os.path.isfile(os.path.join(path_image_label, f))]

    for fichier in fichiers:

      try:

        img = cv.imread(path_image_label + '/' + fichier, cv.IMREAD_GRAYSCALE)

        resized_img = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        list_images.append([resized_img, labels.index(label)])

      except:

        print(fichier)



  return np.array(list_images)
train_data = get_images('../input/chest-xray-pneumonia/chest_xray/chest_xray/train')

val_data = get_images('../input/chest-xray-pneumonia/chest_xray/chest_xray/val')

test_data = get_images('../input/chest-xray-pneumonia/chest_xray/chest_xray/test')
print('{} images in training set'.format(len(train_data)))

print('--- {} NORMAL IMAGES IN TRAINING SET'.format([y for _, y in train_data].count(0)))

print('--- {} PNEUMONIA IMAGES IN TRAINING SET'.format([y for _, y in train_data].count(1)))
print('{} images in validation set'.format(len(val_data)))

print('--- {} NORMAL IMAGES IN VALIDATION SET'.format([y for _, y in val_data].count(0)))

print('--- {} PNEUMONIA IMAGES IN VALIDATION SET'.format([y for _, y in val_data].count(1)))
print('{} images in test set'.format(len(test_data)))

print('--- {} NORMAL IMAGES IN TEST SET'.format([y for _, y in test_data].count(0)))

print('--- {} PNEUMONIA IMAGES IN TEST SET'.format([y for _, y in test_data].count(1)))
l = []

for i in train_data:

    if(i[1] == 0):

        l.append("Normal")

    else:

        l.append("Pneumonia")

sns.set_style('darkgrid')

sns.countplot(l)   
plt.figure(figsize=(10, 10))

for k, i in np.ndenumerate(np.random.randint(train_data.shape[0], size=9)):

    ax = plt.subplot(3, 3, k[0] + 1)

    plt.imshow(train_data[i][0], cmap='gray')

    plt.title(labels[train_data[i][1]])

    plt.axis("off")
def prepare_data(data):

    x = []

    y = []

    

    for feature, label in data:

      x.append(feature)

      y.append(label)



    x = (np.array(x) / 255).reshape(-1,IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)

    y = np.array(y)

        

    return x, y
x_train, y_train = prepare_data(train_data)

x_val, y_val = prepare_data(val_data)

x_test, y_test = prepare_data(test_data)
print('The new shape of images is {}'.format(x_train[0].shape))

print('Number of images train is {}'.format(x_train.shape[0]))
datagen = ImageDataGenerator(

    rotation_range = 20, 

    zoom_range = 0.2, 

    width_shift_range=0.15,  

    height_shift_range=0.15,

    shear_range=0.15,

    horizontal_flip = False,  

    vertical_flip=False,

    fill_mode="nearest")
class Model2:

  @staticmethod

  def build(width, height, depth, classes, reg):

    inputShape = (height, width, depth)



    # Input

    input_layer = Input(shape=inputShape, name='Image_input')

    conv_layer_1 = Conv2D(32, (5, 5), strides=(2, 2), padding="same",

                   kernel_regularizer=reg, input_shape=inputShape,

                   name='Conv_1')(input_layer)

    batch_norm_1 = BatchNormalization(name='Batch_1')(conv_layer_1)

    dropout_1 = Dropout(0.2, name='Drop_1')(batch_norm_1)



    # Part 1

    conv_layer_2 = Conv2D(64, (3, 3), strides=(2, 2), padding="same",

                   kernel_regularizer=reg, input_shape=inputShape,

                   name='Conv_2_1_1')(dropout_1)

    batch_norm_2 = BatchNormalization(name='Batch_2_1_1')(conv_layer_2)

    dropout_2 = Dropout(0.2, name='Drop_2_1_1')(batch_norm_2)



    conv_layer_3 = Conv2D(128, (3, 3), strides=(2, 2), padding="same",

                   kernel_regularizer=reg, input_shape=inputShape,

                   name='Conv_2_1_2')(dropout_2)

    batch_norm_3 = BatchNormalization(name='Batch_2_1_2')(conv_layer_3)

    dropout_3 = Dropout(0.2, name='Drop_2_1_2')(batch_norm_3)



    conv_layer_4 = Conv2D(256, (5, 5), strides=(2, 2), padding="same",

                   kernel_regularizer=reg, input_shape=inputShape,

                   name='Conv_2_1_3')(dropout_3)

    batch_norm_4 = BatchNormalization(name='Batch_2_1_3')(conv_layer_4)

    max_pool_4 = MaxPool2D((2,2) , strides = 2 , padding = 'same',

                           name='Max_Pool_2_1')(batch_norm_4)

    dropout_4 = Dropout(0.2, name='Drop_2_1_3')(max_pool_4)





    # Part 2

    conv_layer_5 = Conv2D(64, (3, 3), strides=(2, 2), padding="same",

                   kernel_regularizer=reg, input_shape=inputShape,

                   name='Conv_2_2_1')(dropout_1)

    batch_norm_5 = BatchNormalization(name='Batch_2_2_1')(conv_layer_5)

    max_pool_5 = MaxPool2D((2,2) , strides = 2 , padding = 'same',

                           name='Max_Pool_2_2_1')(batch_norm_5)

    dropout_5 = Dropout(0.2, name='Drop_2_2_1')(max_pool_5)



    conv_layer_6 = Conv2D(128, (5, 5), strides=(2, 2), padding="same",

                   kernel_regularizer=reg, input_shape=inputShape,

                   name='Conv_2_2_2')(dropout_5)

    batch_norm_6 = BatchNormalization(name='Batch_2_2_2')(conv_layer_6)

    max_pool_6 = MaxPool2D((2,2) , strides = 2 , padding = 'same',

                           name='Max_Pool_2_2_2')(batch_norm_6)

    dropout_6 = Dropout(0.2, name='Drop_2_2_2')(max_pool_6)



    # Part 3

    conv_layer_7 = Conv2D(64, (7, 7), strides=(2, 2), padding="same",

                   kernel_regularizer=reg, input_shape=inputShape,

                   name='Conv_2_3')(dropout_1)

    batch_norm_7 = BatchNormalization(name='Batch_2_3')(conv_layer_7)

    max_pool_7 = MaxPool2D((4, 4) , strides = 4 , padding = 'same',

                           name='Max_Pool_2_3_1')(batch_norm_7)

    max_pool_8= MaxPool2D((2, 2) , strides = 2 , padding = 'same',

                          name='Max_Pool_2_3_2')(max_pool_7)

    dropout_7 = Dropout(0.2, name='Drop_2_3')(max_pool_8)



    # Concatenate layer

    merged_layer = Concatenate(name='Concat')([dropout_4, dropout_6, dropout_7])

    

    # fully-connected layer

    flatten = Flatten(name='Flatten')(merged_layer)

    dense_1 = Dense(256, activation='relu', name='FC1')(flatten)

    dense_2 = Dense(128, activation='relu', name='FC2')(dense_1)



    final_layer = Dense(classes, activation='sigmoid')(dense_2)

   

    # return the constructed network architecture

    return Model(inputs=input_layer, outputs=final_layer, name="Model_2")
model = Model2.build(width=IMG_WIDTH, height=IMG_HEIGHT, depth=IMG_DEPTH, classes=1, reg=l2(0.0005))
model.summary()
plot_model(model, 'model2.png', show_shapes=True)
opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4 / EPOCH)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
history = model.fit(datagen.flow(x_train, y_train, batch_size = BATCH_SIZE) ,epochs = EPOCH , validation_data = (x_val, y_val),

                    steps_per_epoch=len(x_train) // BATCH_SIZE)
evaluation = model.evaluate(x_test,y_test, verbose=0)

print("Loss of the model is - " , evaluation[0])

print("Accuracy of the model is - " , evaluation[1]*100 , "%")
# plot the training loss

N = EPOCH

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, N), history.history['loss'], label="train_loss")

plt.plot(np.arange(0, N), history.history['val_loss'], label="val_loss")

plt.title("Training Loss on Dataset")

plt.xlabel("Epoch #")

plt.ylabel("Loss")

plt.legend(loc="lower left")

plt.savefig("plot.png")
# plot the training loss and accuracy

N = EPOCH

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, N), history.history['accuracy'], label="train_acc")

plt.plot(np.arange(0, N), history.history['val_accuracy'], label="val_acc")

plt.title("Training Accuracy on Dataset")

plt.xlabel("Epoch #")

plt.ylabel("Accuracy")

plt.legend(loc="lower left")

plt.savefig("plot.png")
predictions = np.array(tf.greater(model.predict(x_test), .5))

predictions = predictions.reshape(1,-1)[0]

predictions = np.array([0 if i == False else True for i in predictions])

predictions[:10]
print(classification_report(y_test, predictions, target_names = ['Normal (Class 0)','Pneumonia (Class 1)']))
cm = confusion_matrix(y_test, predictions)

cm
plt.figure(figsize = (10,10))

sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='',xticklabels = labels,yticklabels = labels)