# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow import keras



import matplotlib.pyplot as plt

import matplotlib.cm as cm
train_data_raw = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data_raw = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



print('train_data_shape: ' + str(train_data_raw.shape))

print('test_data_shape: ' + str(test_data_raw.shape))
def pre_process_data(dataset, has_label=True):

    if has_label:

        data = dataset.iloc[:,1:].values

    else:

        data = dataset.iloc[:,:].values

        

    data = data.astype(np.float)

    

    data = np.multiply(data, 1.0 / 255.0)

    data = data.reshape(data.shape[:1] + (28, 28, 1))

#     data = data.reshape(dataset.shape[0], 28, 28)

        

    return data



train_images = pre_process_data(train_data_raw)



print(train_images.shape)

print(train_images)
def pre_process_labels(data):

    labels_flat = data.iloc[:,0].values.ravel()

    num_classes = np.unique(labels_flat).shape[0]

    

    num_labels = labels_flat.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_flat.ravel()] = 1



    return labels_one_hot.astype(np.uint8)



train_labels = pre_process_labels(train_data_raw)



print(train_labels.shape)

print(train_labels)
def display(img):

    height = width = int(img.shape[0])



    # (784) => (28,28)

    one_image = img.reshape(width, height)

    

    plt.axis('off')

    plt.imshow(one_image, cmap=cm.binary)



display(train_images[7])
print("Images shape: " + str(train_images.shape))

print("Labels shape: " + str(train_labels.shape))
def show_history(history):

    fig, ax = plt.subplots(1, 2, figsize=(15,5))

    ax[0].set_title('loss')

    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")

    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")

    ax[1].set_title('acc')

    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")

    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation acc")

    ax[0].legend()

    ax[1].legend()
def fit(train_images, train_labels, config):

    NUM_EPOCHS = config['NUM_EPOCHS']

    BATCH_SIZE = config['BATCH_SIZE']

    LEARNING_RATE = config['LEARNING_RATE']

    DROPOUT_RATE = config['DROPOUT_RATE']

    NETWORK_WIDTH = config['NETWORK_WIDTH']

    

    initializer = tf.keras.initializers.GlorotNormal()

    

    model = keras.models.Sequential([

        keras.layers.Conv2D(32,kernel_size=(3, 3),activation='swish',kernel_initializer=initializer, input_shape=(28,28,1), use_bias=True),

        keras.layers.Conv2D(32,kernel_size=(3,3), activation='swish',kernel_initializer=initializer, use_bias=True),

        keras.layers.MaxPooling2D(pool_size=(2,2)),

        keras.layers.Dropout(DROPOUT_RATE),

        keras.layers.Conv2D(64,(3, 3),activation='swish', padding='same', use_bias=True),

        keras.layers.MaxPooling2D(pool_size=(2,2)),

        keras.layers.Dropout(DROPOUT_RATE),

        keras.layers.Conv2D(512,(3, 3),activation='swish', padding='same', use_bias=True),

        keras.layers.MaxPooling2D(pool_size=(2,2)),

        keras.layers.Dropout(DROPOUT_RATE),

        keras.layers.Conv2D(256,(2, 2),activation='swish', padding='same', use_bias=True),

        keras.layers.MaxPooling2D(pool_size=(2,2)),

        keras.layers.Dropout(DROPOUT_RATE),

        keras.layers.Flatten(),

        keras.layers.Dense(128,activation='swish', kernel_regularizer =tf.keras.regularizers.l1(l=0.01)),

#         keras.layers.BatchNormalization(),

        keras.layers.Dropout(DROPOUT_RATE),

        keras.layers.Dense(10, activation=tf.nn.softmax),

    ])



    



    

    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    loss = tf.keras.losses.CategoricalCrossentropy()



    model.compile(

        optimizer=opt,

        loss=loss,

        metrics=['accuracy']

    )



    

    val_size = int(train_images.shape[0] * 0.2)



    val_images = train_images[:val_size,:, :]

    val_labels = train_labels[:val_size,:]



    train_images = train_images[val_size:,:, :]

    train_labels = train_labels[val_size:,:]



    hist = model.fit(

        x=train_images, y=train_labels,

        epochs=NUM_EPOCHS,

        batch_size=BATCH_SIZE,

        validation_steps=10,

        validation_data=(val_images, val_labels),

        verbose=1

    )

    

    show_history(hist)

    

    loss, acc = model.evaluate(val_images, val_labels)

    

    return model, loss, acc
config = {

    'BATCH_SIZE': 128,

    'LEARNING_RATE': 0.001,

    'DROPOUT_RATE': 0.4,

    'NUM_EPOCHS': 50,

    'NETWORK_WIDTH': 512

}



model, loss, acc = fit(train_images, train_labels, config)



print('Accuracy: ' + str(acc * 100) + '%')
test_data = pre_process_data(test_data_raw, False)



test_data.shape
predictions = model.predict(test_data)
predictions
# submissions for Kaggle

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": np.argmax(predictions, axis=1)})

submissions.to_csv("my_submissions.csv", index=False, header=True)