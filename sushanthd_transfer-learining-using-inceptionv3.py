import os

os.listdir('../input/input/input/keras-pretrained-models/')
import os

cache_dir = os.path.expanduser(os.path.join('~', '.keras'))

if not os.path.exists(cache_dir):

    os.makedirs(cache_dir)

models_dir = os.path.join(cache_dir, 'models')

if not os.path.exists(models_dir):

    os.makedirs(models_dir)
!cp ../input/input/input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 ~/.keras/models/
print(os.listdir('../input/input/input/plant-seedlings-classification/train/'))
import fnmatch

import os

import numpy as np

import pandas as pd

from keras.applications.vgg16 import preprocess_input

from keras.applications.inception_v3 import preprocess_input

from keras.preprocessing import image

np.random.seed(21)



path = '../input/input/input/plant-seedlings-classification/train/'

train_label = []

train_img = []

label2num = {'Loose Silky-bent':0, 'Charlock':1, 'Sugar beet':2, 'Small-flowered Cranesbill':3,

             'Common Chickweed':4, 'Common wheat':5, 'Maize':6, 'Cleavers':7, 'Scentless Mayweed':8,

             'Fat Hen':9, 'Black-grass':10, 'Shepherds Purse':11}

for i in os.listdir(path):

    label_number = label2num[i]

    new_path = path+i+'/'

    for j in fnmatch.filter(os.listdir(new_path), '*.png'):

        temp_img = image.load_img(new_path+j, target_size=(200,200))

        train_label.append(label_number)

        temp_img = image.img_to_array(temp_img)

        train_img.append(temp_img)



train_img = np.array(train_img)



train_y=pd.get_dummies(train_label)

train_y = np.array(train_y)

train_img=preprocess_input(train_img)



print('Training data shape: ', train_img.shape)

print('Training labels shape: ', train_y.shape)
def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision





def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall





def fscore(y_true, y_pred):

    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:

        return 0



    p = precision(y_true, y_pred)

    r = recall(y_true, y_pred)

    f_score = 2 * (p * r) / (p + r + K.epsilon())

    return f_score
# import keras

# from keras.models import Sequential,Model

# from keras.layers import Dense, Dropout, Flatten, Activation

# from keras.layers import Conv2D, MaxPooling2D

# from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate

# from keras.layers.normalization import BatchNormalization

# from keras.applications.inception_v3 import InceptionV3

# from keras.preprocessing.image import ImageDataGenerator

# from keras import backend as K

# from keras import optimizers, losses, activations, models



# base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(200,200,3))

# #model.summary()

# base_model.trainable = False



# add_model = Sequential()

# add_model.add(base_model)

# add_model.add(GlobalAveragePooling2D())

# add_model.add(Dropout(0.5))

# add_model.add(Dense(12, activation='softmax'))



# model = add_model

# model.compile(loss='categorical_crossentropy', 

#               optimizer=optimizers.SGD(lr=1e-4, 

#                                        momentum=0.9),

#               metrics=['accuracy',fscore])

# model.summary()
import keras

from keras.models import Sequential,Model

from keras.layers import Dense, Dropout, Flatten, Activation

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing.image import ImageDataGenerator



def inceptionv3_model(num_classes=None):



    model = InceptionV3(weights='imagenet', include_top=False,input_shape=(200,200,3))

    model.layers.pop()

    model.layers.pop()

    model.layers.pop()



    model.outputs = [model.layers[-1].output]



    model.layers[-2].outbound_nodes= []

    x=Conv2D(256, kernel_size=(2,2),strides=2)(model.output)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)    

    x=Conv2D(128, kernel_size=(2,2),strides=1)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x=Conv2D(64, kernel_size=(2,2),strides=2)(model.output)

    x = BatchNormalization()(x)

    x = Activation('relu')(x) 

    x=Flatten()(x)

    x=Dense(num_classes, activation='softmax')(x)



    model=Model(model.input,x)



    for layer in model.layers[:46]:



        layer.trainable = False





    return model
from keras import backend as K

num_classes=12

model = inceptionv3_model(num_classes)

model.compile(optimizer="nadam", loss='categorical_crossentropy', metrics=['accuracy',fscore])

# model.compile(loss='categorical_crossentropy', 

#               optimizer=optimizers.SGD(lr=1e-4, 

#                                        momentum=0.9),

#               metrics=['accuracy'])

model.summary()
#Split training data into rain set and validation set

from sklearn.model_selection import train_test_split

X_train, X_valid, Y_train, Y_valid=train_test_split(train_img,train_y,test_size=0.1, random_state=42)



#Data augmentation

'''from keras.preprocessing.image import ImageDataGenerator

gen_train = ImageDataGenerator( 

    rotation_range=30,

    width_shift_range=0.2,

   height_shift_range=0.2,

    horizontal_flip=True,

    vertical_flip=True



)

gen_train.fit(X_train)



#Train model

from keras.callbacks import ModelCheckpoint

epochs = 10

batch_size = 32

model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)



model.fit_generator(gen_train.flow(X_train, Y_train, batch_size=batch_size, shuffle=True), 

                    steps_per_epoch=(X_train.shape[0]//(4*batch_size)), 

                    epochs=epochs, 

                    validation_data=(X_valid,Y_valid),

                    callbacks=[model_checkpoint],verbose=1)

'''

from keras.callbacks import ModelCheckpoint

epochs = 10

batch_size = 32

# model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)



model_checkpoint = ModelCheckpoint('./model62.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5',

                           monitor='val_loss',

                             verbose=1,

                            save_best_only=True,

                            mode='min',

                             save_weights_only=False)





model.fit(X_train,Y_train,

          batch_size=128,

          epochs=20,

          verbose=1, shuffle=True, validation_data=(X_valid,Y_valid), callbacks=[model_checkpoint])
# train_img= train_img.reindex(np.random.permutation(train_img.index))

# train_y 

# #test_proportion of 3 means 1/3 so 33% test and 67% train

# def shuffle(matrix, target, test_proportion):

#     ratio = int(matrix.shape[0]/test_proportion) #should be int

#     X_train = matrix[ratio:,:]

#     X_test =  matrix[:ratio,:]

#     Y_train = target[ratio:,:]

#     Y_test =  target[:ratio,:]

#     return X_train, X_test, Y_train, Y_test



# X_train, X_test, Y_train, Y_test = shuffle(X, Y, 3)



#np.random.shuffle(train_img)
from keras.models import load_model

model=load_model("model62.14-0.59-0.85.h5",custom_objects={'fscore': fscore})
# import matplotlib.pyplot as plt

# def plot_model(model):

#     plots = [i for i in model.history.history.keys() if i.find('val_') == -1]

#     plt.figure(figsize=(10,10))



#     for i, p in enumerate(plots):

#         plt.subplot(len(plots), 2, i + 1)

#         plt.title(p)

#         plt.plot(model.history.history[p], label=p)

#         plt.plot(model.history.history['val_'+p], label='val_'+p)

#         plt.legend()



#     plt.show()

    

# plot_model(model)
from sklearn.metrics import classification_report, confusion_matrix

labelNames=['Loose Silky-bent', 'Charlock', 'Sugar beet', 'Small-flowered Cranesbill',

             'Common Chickweed', 'Common wheat', 'Maize', 'Cleavers', 'Scentless Mayweed',

             'Fat Hen', 'Black-grass', 'Shepherds Purse']

preds = model.predict(X_valid)

print("[INFO] evaluating network...")

print(classification_report(Y_valid.argmax(axis=1), preds.argmax(axis=1),

	target_names=labelNames))

print("The confusion matrix:")

print(confusion_matrix(Y_valid.argmax(axis=1), preds.argmax(axis=1)))