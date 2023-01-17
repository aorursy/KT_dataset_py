import tensorflow as tf

import numpy as np

import os 

import pickle

import pandas as pd

from tensorflow import keras

CIFAR10_DIR = "/kaggle/input/cifar10"

print(os.listdir(CIFAR10_DIR))

with open(os.path.join(CIFAR10_DIR,"data_batch_1"),"rb") as f:

    data = pickle.load(f,encoding="bytes")

    print(type(data))

    print(data.keys())

    print(data[b"batch_label"])

    print(data[b"labels"][:2])

    print(data[b"data"][1].shape)

    print(data[b"filenames"][1])
import matplotlib.pyplot as plt

class_name = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

label = data[b"labels"][2]

print("This is a %s"%class_name[label])

img = data[b"data"][2].reshape(3,32,32).transpose(1,2,0)

plt.imshow(img)

plt.show()
def make_data(filename):

    CIFAR10_DIR = "/kaggle/input/cifar10"

    data_all = []

    labels_all = []

    for filename in filename:

        with open(os.path.join(CIFAR10_DIR,filename),"rb") as f:

            data = pickle.load(f,encoding="bytes")

            data_all.append(data[b"data"])

            labels_all.append(data[b"labels"])

    return np.vstack(data_all), np.hstack(labels_all)

#get the train data

data_all, labels_all = make_data(os.listdir(CIFAR10_DIR)[:5])

print(data_all.shape)

print(labels_all.shape)

#get the test data

test_data, test_labels = make_data(os.listdir(CIFAR10_DIR)[-1:])

print(test_data.shape)

print(test_labels.shape)
from sklearn.model_selection import train_test_split

train_data, valid_data, train_labels, valid_labels = train_test_split(data_all, labels_all, random_state=7)

print(train_data.shape)

print(valid_data.shape)

print(train_labels.shape)

print(valid_labels.shape)

def generate_data(data, labels, batch_size):

    data = data.reshape(-1,3,32,32).transpose((0,2,3,1)) #(None, 3072) --> (None, 32, 32, 3)

    data = data / 127.5 - 1 # normalization

    dataset = tf.data.Dataset.from_tensor_slices((data, labels))

    dataset = dataset.repeat()

    dataset = dataset.shuffle(10000)

    dataset = dataset.batch(batch_size)

    return dataset

batch_size = 64

train_dataset = generate_data(train_data, train_labels, batch_size)

valid_dataset = generate_data(valid_data, valid_labels, batch_size)

test_dataset = generate_data(test_data, test_labels, batch_size)
for x,y in train_dataset.take(3):

    print(x.shape,y.shape)
def data_generator(data, labels, shuffle=True):

    data = data.reshape(-1,3,32,32).transpose((0,2,3,1))

    data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,

                                                             rotation_range=40,

                                                             width_shift_range=0.2,

                                                             height_shift_range=0.2,

                                                             shear_range=0.2,

                                                             zoom_range=0.2,

                                                             horizontal_flip=True,

                                                             fill_mode="nearest",)



    data_generator = data_gen.flow(data, labels,                 

                                    batch_size=batch_size,

                                    seed=7,

                                    shuffle=shuffle,)

    return data_generator



traindata_generator = data_generator(train_data, train_labels)

validdata_generator = data_generator(valid_data, valid_labels)

testdata_generator = data_generator(test_data, test_labels, shuffle=False)

print(type(traindata_generator))
def conv_wrapper(layer_channels, model):

    for num, channel in enumerate(layer_channels):

        if num == 0:#judge whether it is the first layer

            model.add(keras.layers.Conv2D(filters=channel, kernel_size = 3, activation = "relu", 

                                          padding="same", input_shape=[img_width, img_height, img_channel]))

        else:

            model.add(keras.layers.Conv2D(filters=channel, kernel_size = 3, activation = "relu", padding="same"))

        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Conv2D(filters=channel, kernel_size = 3, activation = "relu", padding="same"))

        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPool2D(pool_size=2))

    return model

                  

img_width = 32

img_height = 32

img_channel = 3

num_class = 10

layer_channels = [64,128,256]



model = keras.models.Sequential()

model = conv_wrapper(layer_channels, model)

#fully_connected_layer

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(num_class, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])       

#method 1 model:

model1 = model

#method 2 model:

model2 = model
model1.summary()
epochs = 20

history = model1.fit(train_dataset,

          steps_per_epoch=len(train_data) // batch_size,

          epochs=epochs,

          validation_data=valid_dataset,

          validation_steps=len(valid_data) // batch_size)
def plot_learning_curves(history, label, epcohs, min_value, max_value):

    import pandas as pd

    data = {}

    data[label] = history.history[label]

    data['val_'+label] = history.history['val_'+label]

    pd.DataFrame(data).plot(figsize=(8, 5))

    plt.grid(True)

    plt.axis([0, epochs, min_value, max_value])

    plt.show()

    

plot_learning_curves(history, 'accuracy', epochs, 0.2, 1)

plot_learning_curves(history, 'loss', epochs, 0, 2)
model1.evaluate(testdata_generator, steps=len(test_data)//batch_size )
model1.evaluate(test_dataset, steps=len(test_data)//batch_size )
epochs = 20

history = model2.fit(traindata_generator,

          steps_per_epoch=len(train_data) // batch_size,

          epochs=epochs,

          validation_data=validdata_generator,

          validation_steps=len(valid_data) // batch_size)
def plot_learning_curves(history, label, epcohs, min_value, max_value):

    import pandas as pd

    data = {}

    data[label] = history.history[label]

    data['val_'+label] = history.history['val_'+label]

    pd.DataFrame(data).plot(figsize=(8, 5))

    plt.grid(True)

    plt.axis([0, epochs, min_value, max_value])

    plt.show()

    

plot_learning_curves(history, 'accuracy', epochs, 0.4, 1)

plot_learning_curves(history, 'loss', epochs, 0, 2)
# 在使用 method 2 的时候， evaluate 的数据要使用 testdata_generator 或 test_dataset 的， 而不是test_data, 因为test_data还没有进行数据归一化

model2.evaluate(testdata_generator, steps=len(test_data)//batch_size )
model2.evaluate(test_dataset, steps=len(test_data)//batch_size )
test_predict = model1.predict(testdata_generator, steps=len(test_data)//batch_size,

                             workers=10,

                            )

print(test_predict.shape)
predict_test=np.argmax(test_predict, axis=1)

print(predict_test.shape)
predict_label = [class_name[index] for index in predict_test]

print(predict_label[:5])
num = np.random.randint(len(test_data))

random_img = test_data[num]

random_label = test_labels[num]

print("The original image is %s"%class_name[test_labels[num]])

img = random_img.reshape(3,32,32).transpose((1,2,0))

plt.imshow(img)

plt.show()

random_img = random_img.reshape(-1,3,32,32).transpose((0,2,3,1))

random_img = random_img * 1. / 225 #记得进行归一化处理，否则测试输入的图片与训练的图片不一样，一个是0-1 一个是0-255

predict_img = class_name[np.argmax(model1.predict(random_img, verbose=1),axis=1)[0]]

print("The predicted result: %s"%predict_img)