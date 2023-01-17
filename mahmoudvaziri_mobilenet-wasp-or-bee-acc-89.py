import numpy as np

import pandas as pd 

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

from tqdm import tqdm

import cv2 as cv

import tensorflow as tf

import matplotlib.pyplot as plt
main_folder = '../input/bee-vs-wasp/kaggle_bee_vs_wasp/'

df = pd.read_csv('../input/bee-vs-wasp/kaggle_bee_vs_wasp/labels.csv')

df = df[df.photo_quality==1]

df.head()
'''

From https://www.kaggle.com/koshirosato/bee-or-wasp-base-line-using-resnet50

'''

for idx in tqdm(df.index):    

    df.loc[idx,'path']=df.loc[idx,'path'].replace('\\', '/') 

    

df.head()
df_test = df[df.is_final_validation==1].reset_index()

df_train = df[df.is_final_validation!=1].reset_index()

df_train.shape,df_test.shape
df.label.value_counts().plot.pie(autopct='%1.1f%%')
'''

From https://www.kaggle.com/koshirosato/bee-or-wasp-base-line-using-resnet50

'''

img_size = 225

def create_datasets(df, img_size):

    imgs = []

    for path in tqdm(df['path']):

        img = cv.imread(main_folder + path)

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        img = cv.resize(img, (img_size,img_size))

        imgs.append(img)

        

    imgs = np.array(imgs, dtype='float32')

    imgs = imgs / 255.0

    df = pd.get_dummies(df['label'])

    return imgs, df





train, df_train = create_datasets(df_train, img_size)

test, df_test = create_datasets(df_test, img_size)
model = Sequential()

model.add(layers.Input(shape=(img_size,img_size,3)))

model.add(tf.keras.applications.MobileNetV2(include_top=False,weights="imagenet"))

model.add(layers. GlobalAveragePooling2D())#BatchNormalization()

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(3,activation='softmax'))

for layer in model.layers[:1]:

    layer.trainable = False

model.summary()
def scheduler(epoch, lr):

    print(model.optimizer.lr.numpy())

    if epoch < 5:

        return lr

    else:

        return lr * tf.math.exp(-0.1)

lr_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('Best_model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train,df_train,batch_size=128,epochs=25,validation_split=0.1,callbacks=[checkpoint_cb,lr_cb],verbose=1)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(acc))



fig, axes = plt.subplots(1, 2, figsize=(15,5))



axes[0].plot(epochs, acc, 'r-', label='Training Accuracy')

axes[0].plot(epochs, val_acc, 'b--', label='Validation Accuracy')

axes[0].set_title('Training and Validation Accuracy')

axes[0].legend(loc='best')



axes[1].plot(epochs, loss, 'r-', label='Training Loss')

axes[1].plot(epochs, val_loss, 'b--', label='Validation Loss')

axes[1].set_title('Training and Validation Loss')

axes[1].legend(loc='best')



plt.show()
model.evaluate(test,df_test,verbose=0)
# del model

# import gc

# gc.collect()