import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from matplotlib.image import imread

import os

import cv2

import random

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.preprocessing.image import img_to_array

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
label=[]

path=("../input/dogs-vs-cats/train/train")

filenames= os.listdir(path) 

for filename in filenames :

    

    filename=filename.split('.')[0]

    if filename=='cat' :

        label.append('0')

    else :

        label.append('1')

        



df = pd.DataFrame({

    'filename': filenames,

    'category': label

})

    

df.info()

print(df["category"])
from keras.models import Sequential

train_df,validate_df=train_test_split(df,test_size=0.20,random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)



from keras.models import Sequential

model=Sequential()



model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu',input_shape=(200,200,3)))

model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes



model.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')

model.summary()
learning_rate_reduction=ReduceLROnPlateau(monitor='val_acc', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.0001)

earlystop=EarlyStopping(patience=10)
callbacks=[earlystop,learning_rate_reduction]

epochs=35

batch_size=100
train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_generator = train_datagen.flow_from_dataframe(  train_df, 

    "../input/dogs-vs-cats/train/train/", 

    x_col='filename',

    y_col='category',

    target_size=(200,200),

    class_mode='categorical',

    batch_size=100

    

)

                                          

                                    





                                             

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    "../input/dogs-vs-cats/train/train/", 

    x_col='filename',

    y_col='category',

    target_size=(200,200),

    class_mode='categorical',

    batch_size=batch_size

)
history=model.fit_generator(train_generator,epochs=epochs, validation_data=validation_generator,

    validation_steps=validate_df.shape[0]//batch_size,

    steps_per_epoch=train_df.shape[0]//batch_size,

    callbacks=callbacks

)
model.save_weights("model.h5")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(history.history['loss'], color='b', label="Training loss")

ax1.plot(history.history['val_loss'], color='r', label="validation loss")

ax1.set_xticks(np.arange(1, epochs, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")

ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

ax2.set_xticks(np.arange(1, epochs, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()



path1="../input/dogs-vs-cats/test/test"

test_filenames=os.listdir(path1)

test_df=pd.DataFrame({

    'filename':test_filenames

})

print(test_df)

"""for filename in filenames:

    image=load_img("../input/dogs-vs-cats/test/test/"+filename,target_size=(200,200))

    plt.imshow(image)

    image=img_to_array(image)

    image=np.expand_dims(image,axis=0)

    result=model.predict(image)

    print(result)

    result=np.argmax(result)

    print(result

    if result==1:

        prediction='dog'

    else:

        prediction='cat'



    print(prediction)"""



test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "../input/dogs-vs-cats/test/test/", 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=(200,200),

    batch_size=100,

    shuffle=False,

    follow_links=False,

    validate_filenames=False

)

predict=model.predict_generator(test_generator)



predict_df=np.argmax(predict,axis=1)

print(predict_df[100])

test_df["category"]=predict_df
label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

sample_test = test_df.head(18)

sample_test.head()

plt.figure(figsize=(12, 24))

for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img("../input/dogs-vs-cats/test/test/"+filename, target_size=(200,200))

    plt.subplot(6, 3, index+1)

    plt.imshow(img)

    plt.xlabel(filename + '(' + "{}".format(category) + ')' )

plt.tight_layout()

plt.show()