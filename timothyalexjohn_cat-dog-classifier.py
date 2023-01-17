import numpy as np

import pandas as pd

import os

import shutil                     # File_Operation Library

import cv2

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Flatten, GlobalAveragePooling2D, Dropout, Input, Concatenate, BatchNormalization



from tensorflow.keras import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3
shutil.unpack_archive('../input/dogs-vs-cats/train.zip', '/kaggle/working/')

shutil.unpack_archive('../input/dogs-vs-cats/test1.zip', '/kaggle/working/')
train_dir = '/kaggle/working/train/'

test_dir = '/kaggle/working/test1/'
train_df = []

img_list = []

l_list = []



for img in os.listdir(train_dir):

    if img.split('.')[-1]=='jpg':

        img_list.append(train_dir+img)

        l_list.append(img.split('.')[0])



train_df = pd.DataFrame(train_df)

train_df['image'] = img_list

train_df['label'] = l_list



print(train_df.head())
train_datagen = ImageDataGenerator(rescale = 1./255, 

                                   horizontal_flip = True, 

                                   rotation_range = 45, 

                                   shear_range = 19,

                                   zoom_range = 0.2,

                                   validation_split = 0.2)



train_generator = train_datagen.flow_from_dataframe(train_df,

                                                    x_col='image',

                                                    y_col='label',

                                                    target_size = (180, 180), 

                                                    class_mode = 'binary',

                                                    batch_size = 280,

                                                    shuffle = True,

                                                    subset = 'training')



val_generator = train_datagen.flow_from_dataframe(train_df,

                                                  x_col='image',

                                                  y_col='label',

                                                  target_size = (180, 180),

                                                  class_mode = 'binary',

                                                  batch_size = 280,

                                                  shuffle = True,

                                                  subset = 'validation')
inputs = Input((180, 180, 3))

pretrained_model= InceptionV3(include_top= False)

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

                          patience = 5,

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
model.compile(optimizer='RMSprop', loss= 'binary_crossentropy', metrics= ['accuracy'])

history = model.fit_generator(train_generator,

                              epochs = 100,

                              steps_per_epoch = len(train_generator),

                              validation_data = val_generator,

                              validation_steps = len(val_generator),

                              callbacks = callbacks,

                              verbose= 1)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='best')

plt.show()
def predict(path):

    img = cv2.imread(str(path))

    img = cv2.resize(img, (180,180))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    img = np.reshape(img,(1,180,180,3))

    return model.predict(img)
print(train_generator.class_indices)    # Mapping Dictionary
test_df = []

test_list = []



for i in os.listdir(test_dir):

    test_list.append(test_dir +i)

    test_df.append(i.split('.')[0])



target=[]

for path in test_list:

    prediction = predict(path)

    target.append(prediction[0][0])

    

test_df = pd.DataFrame(test_df)

test_df.columns = ['id']

test_df['label'] = target



test_df.sort_values(by=['id'], inplace=True)

test_df.to_csv('submission.csv', index=False)
img_num = 1234   # change me!



path = test_list[img_num]

img=cv2.imread(str(path))

plt.imshow(img)



prediction = predict(path)



if prediction < 0.5:

    print("It's a Cat!")

else:

    print("It's a Dog!")