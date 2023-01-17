import numpy as np

import pandas as pd 

from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt

plt.style.use('ggplot')
# list out the available files in the input path

import os

print(os.listdir("../input"))
!pip install tensorflow-gpu==2.0.0-beta1
!pip install -q tensorflow_hub
import tensorflow as tf

import tensorflow_hub as hub
print(tf.__version__)
train_dir="../input/train/train"

test_dir="../input/test/test"

train = pd.read_csv('../input/train.csv')

sub_file = pd.read_csv("../input/sample_submission.csv")

data_folder = "../input"
train.head()
sub_file.head()
def show_images(directory, df, is_train=True):

    plt.figure(figsize=(15,15))

    for i in range(10):

        n = np.random.choice(df.shape[0], 1)

        plt.subplot(5,5,i+1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(True)

        image = plt.imread(os.path.join(directory, df["id"][int(n)]))

        plt.imshow(image)

        if is_train:

            label = df["has_cactus"][int(n)]

            plt.xlabel(label)

    plt.show()

# train set

show_images(train_dir, train)
# test set

show_images(test_dir, sub_file, is_train=False)
train["has_cactus"].value_counts()
# 90% for train

partial_train = train.sample(frac=0.9)

train.drop(partial_train.index, axis=0, inplace=True)



# 10% for validation

valid = train
partial_train["has_cactus"].value_counts()
valid["has_cactus"].value_counts()
# account for skew in the labeled data

lb = LabelBinarizer()

y_train = lb.fit_transform(partial_train["has_cactus"])

classTotals = y_train.sum(axis=0)

classWeight = classTotals.max() / classTotals
# convert the data-type of the labels to string to make it compatible with

# ImageDataGenerator

partial_train["has_cactus"] = partial_train["has_cactus"].astype("str") 

valid["has_cactus"] = valid["has_cactus"].astype("str") 

sub_file["has_cactus"] = sub_file["has_cactus"].astype("str")
# set up the data augmentation objects

trainAug = tf.keras.preprocessing.image.ImageDataGenerator(

  horizontal_flip=True,

  fill_mode="nearest")



valAug = tf.keras.preprocessing.image.ImageDataGenerator()



# define the ImageNet mean subtraction (in RGB order) and set the

# the mean subtraction value for each of the data augmentation

# objects

mean = np.array([123.68, 116.779, 103.939], dtype="float32")

trainAug.mean = mean

valAug.mean = mean



trainGen = trainAug.flow_from_dataframe(partial_train, directory=train_dir, 

    x_col="id", y_col="has_cactus", target_size=(224, 224), 

    class_mode="categorical", batch_size=64, shuffle=True)



valGen = valAug.flow_from_dataframe(valid, directory=train_dir, 

    x_col="id", y_col="has_cactus", target_size=(224, 224), 

    class_mode="categorical", batch_size=64)



testGen = valAug.flow_from_dataframe(sub_file, directory=test_dir, 

    x_col="id", y_col="has_cactus", target_size=(224, 224), 

    class_mode="categorical", batch_size=64)
# define the input dimension of the KerasLayer and then set its layers to

# trainable to adapt to our dataset

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

feature_extractor_layer = hub.KerasLayer(feature_extractor_url,

                                         input_shape=(224,224,3))

feature_extractor_layer.trainable = True
model = tf.keras.Sequential([

  feature_extractor_layer,

  tf.keras.layers.Dense(2, activation="sigmoid")

])
model.compile(

  optimizer=tf.keras.optimizers.Adam(),

  loss='categorical_crossentropy',

  metrics=['acc'])
H = model.fit_generator(

    trainGen,

    steps_per_epoch=partial_train.shape[0] // 64,

    validation_data=valGen,

    validation_steps=valid.shape[0] // 64,

    epochs=5,

    class_weight=classWeight,

    verbose=1)
def plot_training(H, N):

    plt.style.use("ggplot")

    plt.figure(figsize=(10,8))

    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")

    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

    plt.title("Training Loss and Accuracy")

    plt.xlabel("Epoch #")

    plt.ylabel("Loss/Accuracy")

    plt.legend(loc="upper center")
plot_training(H, 5)
# get the predictions from the network and map 

# the class-labels accordingly

predIdxs = model.predict_generator(testGen,

    steps=(sub_file.shape[0] // 64) + 1)

predIdxs = np.argmax(predIdxs, axis=1)
sub_file.has_cactus = predIdxs

sub_file.to_csv('submission.csv', index=False)