import numpy as np

import pandas as pd

import tensorflow as tf

import skimage.io as sk

import pathlib

import matplotlib.pyplot as plt



import os

import time
DATA_FOLDER=pathlib.Path("/kaggle/input/flower-color-images/flower_images/flower_images")

data_set = pd.read_csv("/kaggle/input/flower-color-images/flower_images/flower_images/flower_labels.csv")
EPOCHS = 30

BS = 15

INIT_LR = 1e-3

TOT_IMG = data_set.count()[0]

TOT_BATCH = int(TOT_IMG/BS)

CLASSES = {0:'phlox',1:'rose',2:'calendula',3:'iris',4:'leucanthemum maximum', 5:'bellflower',6:'viola',7:'rudbeckia laciniata (Goldquelle)',

          8:'peony',9:'aquilegia'}

CLASS_ARR = [0,1,2,3,4,5,6,7,8,9]
print(sk.imread('/kaggle/input/flower-color-images/flower_images/flower_images/0172.png').shape)

sk.imshow('/kaggle/input/flower-color-images/flower_images/flower_images/0172.png')
#decoding Image using TF helper functions.

def decode_img(img):

    # convert the compressed string to a 3D uint8 tensor

    img = tf.image.decode_png(img, channels=3)

    img = tf.image.convert_image_dtype(img, tf.float32)

    img = tf.image.resize(img,[224,224])

    return img
# Extract Label from the image and dataset

def get_label(img_name):

    lab = (data_set.loc[data_set['file']==img_name]['label']).to_string(index=False)

    ilab= np.array([lab]).astype(np.int32)

    ret =np.equal(ilab,CLASS_ARR)

    return ret
#https://www.tensorflow.org/tutorials/load_data/images

# 

def process_image(image_path):

    parts = tf.strings.split(image_path, os.path.sep)    

    image_name = parts[-1]

    label = tf.py_function(func=get_label,inp=[image_name], Tout=tf.bool)

    img = tf.io.read_file(image_path)

    img = decode_img(img)

    return img, label
# Using tf.Data to process dataset for this experiment.

list_ds= tf.data.Dataset.list_files(str(DATA_FOLDER/'*.png'))

labeled_ds = list_ds.map(process_image,num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds = labeled_ds.shuffle(buffer_size=TOT_IMG)

ds = ds.batch(BS)

ds = ds.repeat()

#image_batch, label_batch = next(iter(ds))



#for img,label in labeled_ds.take(1):

#    tf.print("IMAGE:", img.numpy().shape)

#    tf.print("LABEL:",label.numpy())

    
def show_image_batch(image_batch, image_label):

    plt.figure(figsize=(10,10))

    for n in range(BS):

        ax = plt.subplot(3,5,n+1)

        plt.imshow(image_batch[n])

        #print(CLASSES[(np.where(image_label[n])[0][0])])

        plt.title(CLASSES[(np.where(image_label[n])[0][0])])

        plt.axis('off')
image_batch, label_batch = next(iter(ds))

show_image_batch(image_batch.numpy(), label_batch.numpy())
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout

from tensorflow.keras.models import Model

rsntBase = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

#rsntBase.summary()



# define a custom model based on ResNet50

model = rsntBase.output

model = AveragePooling2D(pool_size=(7,7))(model)

model = Flatten(name="flatten")(model)

model = Dense(1024,activation='relu')(model)

model = Dropout(0.5)(model)

model = Dense(1024,activation='relu')(model)

model = Dropout(0.5)(model)

model = Dense(len(CLASS_ARR), activation='softmax')(model)



clfModel = Model(inputs=rsntBase.input, outputs=model)



# freezing all layers except for last Conv block (Conv5)

for _ in rsntBase.layers:

    if not _.name.startswith('conv5_'):

        _.trainable=False

    

clfModel.summary()
# Define Optimizer

optimizer = tf.keras.optimizers.Adam(lr=INIT_LR)
#define training Loss

from tensorflow.keras.losses import categorical_crossentropy


def step_function(X,y):

    with tf.GradientTape() as tape:

        pred = clfModel(X)

        loss = categorical_crossentropy(y, pred,from_logits=True)

        

    grads = tape.gradient(loss, clfModel.trainable_variables)

    optimizer.apply_gradients(zip(grads,clfModel.trainable_variables))

    return loss    
all_loss = []

train_accuracy = []

def train(datasetx, epochs):

    for epoch in range(0, epochs):

        start = time.time()

        epoch_avg_loss = tf.keras.metrics.Mean()

        epoch_accuracy = tf.keras.metrics.CategoricalCrossentropy()

        tds = iter(datasetx)

        for i in range(TOT_BATCH):

            image_batch, label_batch = next(tds)

            loss = step_function(image_batch,label_batch)

            epoch_avg_loss.update_state(loss)

            epoch_accuracy.update_state(label_batch,clfModel(image_batch, training=True))

        all_loss.append(epoch_avg_loss.result().numpy())

        train_accuracy.append(epoch_accuracy.result().numpy())

        if epoch % 2 == 0:

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start), epoch_avg_loss.result().numpy(),epoch_accuracy.result().numpy())            
train(ds, EPOCHS)
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))

fig.suptitle('Training Metrics')



axes[0].set_ylabel("Loss", fontsize=14)

axes[0].plot(all_loss)



axes[1].set_ylabel("Accuracy", fontsize=14)

axes[1].set_xlabel("Epoch", fontsize=14)

axes[1].plot(train_accuracy)

plt.show()
for img,label in ds.take(1):

    plt.figure(figsize=(20,20))

    pred_logits = clfModel(img.numpy(), training=False)

    pred = tf.argmax(pred_logits, axis=1, output_type=tf.int32).numpy()

    print(pred)

    for n in range(BS):

        ax = plt.subplot(3,5,n+1)

        plt.imshow(img[n].numpy())

        plt.title("Actual: {}\nPredicted: {}".format(CLASSES[(np.where(label[n])[0][0])],CLASSES[pred[n]]))

        plt.axis('off')

        

    #tf.print("IMAGE:", img.numpy().shape)

    #tf.print("LABEL:",label.numpy())

    #clfModel(img.numpy(), training=False)

    