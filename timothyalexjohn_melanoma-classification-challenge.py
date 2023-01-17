import numpy as np

import pandas as pd

import os

import cv2



import tensorflow.keras as tk

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.applications import VGG16

from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Flatten,MaxPooling2D, GlobalAveragePooling2D, Dropout, Input, Concatenate, BatchNormalization, Conv2D

from tensorflow.keras import Model

import matplotlib.pyplot as plt
train_dir = '../input/siim-isic-melanoma-classification/jpeg/train/'    

test_dir = '../input/siim-isic-melanoma-classification/jpeg/test/'



train_csv_dir = '../input/siim-isic-melanoma-classification/train.csv'

test_csv_dir = '../input/siim-isic-melanoma-classification/test.csv'



train_csv = pd.read_csv(train_csv_dir)

test_csv = pd.read_csv(test_csv_dir)
train_df = []

train_list = os.listdir(train_dir)



for i in train_list:

    train_df.append(train_dir + i)



train_df = pd.DataFrame(train_df)    

train_df.columns = ['images']

train_df['y'] = train_csv['target']
def hair_remove(image):

    

    # convert image to grayScale

    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    

    # kernel for morphologyEx

    kernel = cv2.getStructuringElement(1,(224,224))

    

    # apply MORPH_BLACKHAT to grayScale image

    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    

    # apply thresholding to blackhat

    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)

    

    # inpaint with original image and threshold image

    final_image = cv2.inpaint(image,threshold,3,cv2.INPAINT_TELEA)

    

    return final_image
train_datagen = ImageDataGenerator(rescale = 1./255, 

                                   horizontal_flip = True, 

                                   vertical_flip = True, 

                                   rotation_range = 45, 

                                   shear_range = 19,

                                   validation_split = 0.15)



train_generator = train_datagen.flow_from_dataframe(train_df,

                                                    x_col='images',

                                                    y_col='y',

                                                    target_size = (224, 224), 

                                                    class_mode = 'raw',

                                                    batch_size = 8,

                                                    shuffle = True,

                                                    subset = 'training')



val_generator = train_datagen.flow_from_dataframe(train_df,

                                                  x_col='images',

                                                  y_col='y',

                                                  target_size = (224, 224),

                                                  class_mode = 'raw',

                                                  batch_size = 8,

                                                  shuffle = True,

                                                  subset = 'validation')


inputs = Input((224, 224, 3))

pretrained_model= VGG16(include_top= False)

x = pretrained_model(inputs)

output1 = GlobalMaxPooling2D()(x)

output2 = GlobalAveragePooling2D()(x)

output3 = Flatten()(x)



outputs = Concatenate(axis=-1)([output1, output2, output3])

outputs = Dropout(0.5)(outputs)

outputs = BatchNormalization()(outputs)

output = Dense(1, activation= 'sigmoid')(outputs)



model = Model(inputs, output)
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback





# autosave best Model

best_model = ModelCheckpoint("model", monitor='val_accuracy', mode='max',verbose=1, save_best_only=True)



earlystop = EarlyStopping(monitor = 'val_accuracy',

                          patience = 3,

                          mode = 'auto',

                          verbose = 1,

                          restore_best_weights = True)



acc_thresh = 0.998



class myCallback(Callback): 

    def on_epoch_end(self, epoch, logs={}): 

        if(logs.get('accuracy') > acc_thresh):   

          print("\nWe have reached %2.2f%% accuracy, so we will stopping training." %(acc_thresh*100))   

          self.model.stop_training = True



callbacks = [myCallback(), best_model, earlystop]
model.compile(optimizer='RMSProp', loss= 'binary_crossentropy', metrics= ['accuracy'])

history = model.fit_generator(train_generator,

                              epochs = 20,

                              steps_per_epoch = len(train_generator),

                              validation_data = val_generator,

                              validation_steps = len(val_generator),

                              callbacks = [callbacks, best_model],

                              verbose= 1)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='best')

plt.show()
test_df = []

test_list = os.listdir(test_dir)



for i in test_list:

    test_df.append(test_dir + i)





test_df = pd.DataFrame(test_df)    

test_df.columns = ['images']
target=[]

for path in test_df['images']:

    img=cv2.imread(str(path))

    img = cv2.resize(img, (224,224))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    img = np.reshape(img,(1,224,224,3))

    prediction = model.predict(img)

    target.append(prediction[0][0])
submission=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')



submission['target']=target



submission.to_csv('submission.csv', index=False)

submission.head()