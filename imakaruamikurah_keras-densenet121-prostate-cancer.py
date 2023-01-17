import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os
os.getcwd()
df_train = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")

df_train.head(10)
train_img_path = "/kaggle/input/prostate-cancer-grade-assessment/train_images"

label_path = "/kaggle/input/prostate-cancer-grade-assessment/train_label_masks"



train_img = [img for img in os.listdir(train_img_path)]

train_label = [label for label in os.listdir(label_path)]



train_img = list(sorted(train_img))

train_label = list(sorted(train_label))
from matplotlib import rcParams

import openslide

import cv2

from IPython.display import display



# rcParams["figure.figsize"] = 15, 15



for i in range(22, 25):

    img = openslide.OpenSlide(train_img_path + "/" + train_img[i])

    display(img.get_thumbnail(size=(600, 400)))

    img.close()

    d = df_train.loc[df_train["image_id"] == train_img[i][:-5]]

    print(f"PROVIDED BY: {d.data_provider.values[0]}")

    print(f"ISUP Grade: {d.isup_grade.values[0]}, Gleason Grade: {d.gleason_score.values[0]}")
import matplotlib



rcParams["figure.figsize"] = 7, 6



for i in range(22, 25):

    mask = openslide.OpenSlide(label_path + "/" + train_label[i])

    mask = mask.get_thumbnail(size=(600, 400))

    mask = np.asarray(mask)

    mask = mask[:,:,0]

    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])



    plt.imshow(mask, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)

    plt.axis('off')

    plt.show()

#     mask.close()
import cv2



SIZE = 200



resized_imgs_path = "../input/panda-resized-train-data-512x512/train_images/train_images"

img_array = []

for i in os.listdir(resized_imgs_path):

    img = resized_imgs_path + "/" + i

    img = cv2.resize(cv2.imread(img), (SIZE, SIZE))

    img_array.append(img)
plt.imshow(img_array[1])

plt.show()
from sklearn.preprocessing import LabelBinarizer



train_y = list(df_train['isup_grade'].values)

lb = LabelBinarizer()

train_y = lb.fit_transform(train_y)
train_y_multi = np.empty(train_y.shape, dtype=train_y.dtype)

train_y_multi[:, 5] = train_y[:, 5]



for i in range(4, -1, -1):

    train_y_multi[:, i] = np.logical_or(train_y[:, i], train_y_multi[:,i+1])
train_y_multi.sum(axis=0)
train_x = np.reshape(img_array, (len(img_array), SIZE, SIZE, 3))
df_test = pd.read_csv("../input/prostate-cancer-grade-assessment/test.csv")

df_test.head(10)
test_img = [img+".tiff" for img in df_test['image_id'].values]
test_img_path = '../input/prostate-cancer-grade-assessment/test_images'

test_x = []



if os.path.exists(test_img_path):

    for i in range(len(test_img)):

        img = test_img_path + "/" + test_img[i]

        img = preprocessing_img(img)

        test_x.append(img)

    test_x = np.reshape(train_img, (len(test_img), SIZE, SIZE, 3))

else:

    test_x = np.random.rand(len(test_img),SIZE, SIZE, 3)
from sklearn.model_selection import train_test_split



train_x, val_x, train_y, val_y = train_test_split(

    train_x, train_y_multi, train_size=0.8, random_state=42)
from keras.applications import DenseNet121



densenet = DenseNet121(

    weights = '../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top=False,

    input_shape=(SIZE, SIZE, 3)

)
from tensorflow.keras.models import Sequential

from tensorflow.keras import layers as ly

from tensorflow.keras.optimizers import Adam



model = Sequential([

    densenet,

    ly.GlobalAveragePooling2D(),

    ly.Dropout(0.8),

    ly.Dense(6, activation="sigmoid")

])



model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.01), metrics=["accuracy"])
model.summary()
from keras.preprocessing.image import ImageDataGenerator



data = ImageDataGenerator(

    zoom_range = 0.15,

    fill_mode="nearest",

    cval=0.,

    horizontal_flip=True,

    vertical_flip=True

)



data = data.flow(train_x, train_y, batch_size=10, seed=42)
history = model.fit_generator(

    data, steps_per_epoch=train_x.shape[0] / 10,

    epochs=10,

    validation_data=(val_x, val_y)

)
acc = history.history["accuracy"]

val_acc = history.history["val_accuracy"]

loss = history.history["loss"]

val_loss = history.history["val_loss"]
plt.plot(acc)

plt.plot(val_acc)

plt.legend(["accuracy", "val_accuracy"])

plt.title("Accuracy and Validation Accuracy throughout epochs")

plt.show()
plt.plot(loss)

plt.plot(val_loss)

plt.title("Loss and Validation Loss throughout epochs")

plt.show()
from random import randint



if os.path.exists(test_img_path):

    test_y = model.predict(test_x)

    test_y = test_y > 0.37757874193797547

    test_y = test_y.astype(int).sum(axis=1) - 1

else:

    test_y = [randint(0, 5) for i in range(3)]



df_test['isup_grade'] = test_y

df_test = df_test[["image_id", "isup_grade"]]

df_test.to_csv("submission.csv", index=False)