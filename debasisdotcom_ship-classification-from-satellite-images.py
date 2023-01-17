import math

import numpy as np 

import pandas as pd 

from PIL import Image, ImageDraw



import matplotlib.pyplot as plt

import seaborn as sns



from keras.applications.mobilenet import MobileNet, preprocess_input

from keras.applications.densenet import DenseNet201



import tensorflow as tf

from keras import Model



from sklearn.model_selection import train_test_split



from tqdm.notebook import tqdm_notebook as tqdm



from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback, LearningRateScheduler



import random

import cv2
train_img_dir = '../input/airbus-ship-detection/train_v2/'

train_seg_csv = '../input/airbus-ship-detection/train_ship_segmentations_v2.csv'

test_img_dir = '../input/airbus-ship-detection/test_v2'

traincsv = pd.read_csv('../input/airbus-ship-detection/train_ship_segmentations_v2.csv')
traincsv.head()
c=[]

for i in (traincsv["EncodedPixels"].notnull()):



    if i==True:

        c.append(1)

    else:

        c.append(0)

        

traincsv["class"]=c



traincsv_unique = traincsv.drop_duplicates(subset=['ImageId'], keep='first')



print(traincsv_unique.head())

print("\n Shape of the Dataframe:",traincsv_unique.shape)
traincsv_unique = traincsv_unique.sort_values(by = ["class"])

traincsv_unique.reset_index(drop = True, inplace = True)



traincsv_unique = pd.concat([traincsv_unique.loc[:4999], traincsv_unique.loc[187556:]])
traincsv_unique["class"].value_counts()
IMAGE_SIZE = 128

paths = traincsv_unique["ImageId"]
batch_images = np.zeros((len(traincsv_unique["ImageId"]), IMAGE_SIZE, IMAGE_SIZE,3), dtype=np.float32)



for i, f in tqdm(enumerate(paths)):

  #print(f)

  img = Image.open(train_img_dir+f)

  img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

  img = img.convert('RGB')

  batch_images[i] = preprocess_input(np.array(img, dtype=np.float32))
batch_images.shape
np.save("D:\\Resume",batch_images)
np.save("class",y)
# batch_images1=batch_images.flatten()

# batch_images1=batch_images.swapaxes(1, 2).reshape(10000*128, 128*3)



# from numpy import savetxt

# savetxt('batch_images.csv', batch_images1, delimiter=',')
y = np.array(traincsv_unique["class"])

print(y)
x_train_data , X_val, y_train_data , y_val = train_test_split(batch_images, y, test_size=0.2, random_state=42)
# model = DenseNet201(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),include_top=False, weights='imagenet', classes=2)





# for layers in model.layers:

#   layers.trainable = False



# x=model.layers[-1].output

# # x=tf.keras.layers.Dense(1024,activation='relu')(x)  

# #x=tf.keras.layers.Dense(512,activation='relu')(x) 

# x=tf.keras.layers.Flatten()(x)

# # x=tf.keras.layers.Dense(128,activation='tanh')(x)

# # x=tf.keras.layers.Dropout(0.4)(x)

# x=tf.keras.layers.Dense(64,activation='tanh')(x)

# x=tf.keras.layers.Dropout(0.4)(x)

# preds=tf.keras.layers.Dense(1,activation='sigmoid')(x) 





# model = Model(inputs = model.inputs, outputs = preds)
ALPHA = 1.0



def schedule(epoch, lr):

        if epoch < 10:

            return lr

        else:

            return lr * tf.math.exp(-0.1)



tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)



# instantiating the model in the strategy scope creates the model on the TPU

with tpu_strategy.scope():

    model = MobileNet(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3), include_top=False, alpha=ALPHA)



    for layers in model.layers:

      layers.trainable = False



    x=model.layers[-1].output

    x=tf.keras.layers.Flatten()(x)

    x=tf.keras.layers.BatchNormalization()(x)

    x=tf.keras.layers.Dense(64,activation='relu')(x)

    x=tf.keras.layers.Dropout(0.6)(x)

    preds=tf.keras.layers.Dense(1,activation='sigmoid')(x) 





    model = Model(inputs = model.inputs, outputs = preds)



    model.compile(loss='binary_crossentropy',

                 optimizer='adam',

                 metrics=['binary_accuracy'])
model.summary()
stop = EarlyStopping(monitor='val_iou', patience=5, mode="max" )

learning_rate = LearningRateScheduler(schedule)

reduce_lr = ReduceLROnPlateau(monitor='val_iou',factor=0.2,patience=5, min_lr=1e-7, verbose=1, mode="max" )



model.fit(x_train_data,

          y_train_data,

          batch_size=64,

          epochs=20,

          callbacks=[stop,reduce_lr,learning_rate],

          validation_data=(X_val, y_val))
predictions = np.round(np.squeeze(model.predict(X_val)))

predictions
i=random.randint(1,1500)



plt.imshow(X_val[i][:, :, 0],cmap='gray')

print("For {}th image:".format(i))

print("\tThe actual label class: ",y_val[i])

print("\tThe predicted label class: ",int(predictions[i]))
unscaled = cv2.imread("../input/airbus-ship-detection/test_v2/000f7d875.jpg")



image_height, image_width, _ = unscaled.shape

image = cv2.resize(unscaled,(IMAGE_SIZE,IMAGE_SIZE))

feat_scaled = preprocess_input(np.array(image, dtype=np.float32))

print("The predicted label",np.round(np.squeeze(model.predict(x = np.array([feat_scaled])))))

plt.imshow(unscaled)