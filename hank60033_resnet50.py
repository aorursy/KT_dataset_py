import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

from sklearn.model_selection import train_test_split



import tensorflow as tf

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Input, Dropout

from keras.models import Sequential, Model

from keras import layers

from keras import regularizers

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

from sklearn.metrics import cohen_kappa_score



import os

print([f for f in os.listdir("../input") if not f.startswith('.')])
train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
train.id_code = train.id_code.apply(lambda x: x + ".png")

test.id_code = test.id_code.apply(lambda x: x + ".png")

train['diagnosis'] = train['diagnosis'].astype('str')
def crop_image_from_gray(img, tol=7):

    

    

    

    if img.ndim == 2:

        mask = img > tol

        return img[np.ix_(mask.any(1),mask.any(0))]



    elif img.ndim == 3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img > tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img 

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img



def preprocess_image(image, sigmaX=10):

    

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (300, 300))

    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)

    return image
train_datagen = ImageDataGenerator(rescale=1. / 128, 

                                         validation_split=0.2,

                                         horizontal_flip=True,

                                         width_shift_range=0.2,

                                         height_shift_range=0.2,

                                         rotation_range=40, 

                                         zoom_range=0.15, 

                                         shear_range=0.15,

                                         preprocessing_function=preprocess_image,

                                         fill_mode='nearest')



train_generator = train_datagen.flow_from_dataframe(dataframe=train,

                                                    directory="../input/aptos2019-blindness-detection/train_images/",

                                                    x_col="id_code",

                                                    y_col="diagnosis",

                                                    batch_size=12,

                                                    class_mode="categorical",

                                                    target_size=(300, 300),

                                                    subset='training',

                                                    shuffle=True)

                                                    

valid_generator = train_datagen.flow_from_dataframe(dataframe=train,

                                                    directory="../input/aptos2019-blindness-detection/train_images/",

                                                    x_col="id_code",

                                                    y_col="diagnosis",

                                                    batch_size=12,

                                                    class_mode="categorical",

                                                    target_size=(300, 300),

                                                    subset='validation',

                                                    shuffle=True)  
conv_base = ResNet50(weights='../input/resnet50-data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False,input_shape=(300, 300, 3)) 

conv_base.summary()
model = Sequential() 

model.add(conv_base) 

model.add(GlobalAveragePooling2D())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(256))

model.add(layers.BatchNormalization())

model.add(layers.Activation('relu'))

model.add(layers.Dense(5, activation='softmax'))
conv_base.trainable = False

model.summary()
learning_rate = 1E-3 



model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.RMSprop(lr=learning_rate),

              metrics=['acc'])
history = model.fit_generator(train_generator,\

                              steps_per_epoch=len(train)*0.8//12,\

                              epochs=2)

#                               validation_data=valid_generator,\

#                              validation_steps=len(train)*0.2//12)  
def get_preds_and_labels(model, generator):

    preds = []

    labels = []

    for _ in range(int(np.ceil(generator.samples / 12))):

        x, y = next(generator)

        preds.append(np.argmax(model.predict(x),axis=1))

        for i in y:

          labels.append(np.argmax(i))

        

    return np.concatenate(preds).ravel(), np.asarray(labels)
class Metrics(Callback):

    

    def on_train_begin(self, logs={}):

        

        self.val_kappas = []



    def on_epoch_end(self, epoch, logs={}):

        

        

        y_pred, labels = get_preds_and_labels(model, valid_generator)

        #y_pred = np.rint(y_pred).astype(np.uint8).clip(0, 4)

        

        _val_kappa = cohen_kappa_score(labels, y_pred,labels=[0,1,2,3,4], weights='quadratic')

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {round(_val_kappa, 4)}")

        if _val_kappa == max(self.val_kappas):

            print("Validation Kappa has improved. Saving model.")

            self.model.save('resnet50-3.h5')

        return
for layer in model.layers:

    layer.trainable = True

    

kappa_metrics = Metrics()

es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)

rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=3, min_lr=1e-6)



callback_list = [kappa_metrics,es, rlrop]

learning_rate = 1E-4 # to be tuned!



model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.RMSprop(lr=learning_rate),

              metrics=['acc'])

model.summary()
history_finetunning = model.fit_generator(train_generator,

                              steps_per_epoch=len(train)*0.8//12,

                              validation_data=valid_generator,

                              validation_steps=len(train)*0.2//12,

                              epochs=30,

                              callbacks=callback_list)

#model_res.save('resnet50-1.h5')
acc = history_finetunning.history['acc']

val_acc = history_finetunning.history['val_acc']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'r', label='Validation acc')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
test_datagen = ImageDataGenerator(rescale=1. / 128,preprocessing_function=preprocess_image)



test_generator = test_datagen.flow_from_dataframe(dataframe=test,

                                                  directory="../input/aptos2019-blindness-detection/test_images/",

                                                  x_col="id_code",

                                                  target_size=(300, 300),

                                                  batch_size=1,

                                                  shuffle=False,                                                  

                                                  class_mode=None)
y_test=np.argmax(model.predict(test_generator), axis=1)
filenames = test_generator.filenames

results = pd.DataFrame({'id_code':filenames, 'diagnosis':y_test})

results['id_code'] = results['id_code'].map(lambda x: str(x)[:-4])

results.to_csv('submission.csv',index=False)

results.head(10)