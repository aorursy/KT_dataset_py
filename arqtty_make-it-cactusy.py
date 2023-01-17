# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
train_img_dir = "../input/train/train/"

train_img_pathes = [train_img_dir + fpath for fpath in sorted(os.listdir(train_img_dir))]



df = pd.read_csv("../input/train.csv")



train_img_pathes[:5]
# Check corresponding between labels and imgs

lst = sorted(os.listdir(train_img_dir))

err = False



for i, idx in enumerate(df["id"]):

    if idx != lst[i]:

        print("mismatch after %d iterations" % i)

        err = True

        break



if not err:

    print("1:1 corresponding between train_img_pathes and df labels")
img_size = 32





def read_and_prep_images(img_paths, img_height=img_size, img_width=img_size):

    # to avoid OSError tooManyOpenedFiles I used batch loading

    img_load_batch_size = 900

    output = None

    

    for i in range(0, len(img_paths), img_load_batch_size):

        print("process batch %d" % i)

        tmp_imgs =  [load_img(img_path, target_size=(img_height, img_width)) 

                     for img_path 

                     in img_paths[i:i+img_load_batch_size]]

        tmp_img_array = np.array([img_to_array(img) for img in tmp_imgs])

        

        if type(output) != np.ndarray:

            output = preprocess_input(tmp_img_array)

        else:

            output = np.vstack((output, preprocess_input(tmp_img_array)))

        

    return(output)





train_imgs = read_and_prep_images(train_img_pathes)

num_classes = 2

out_y = keras.utils.to_categorical(df["has_cactus"], num_classes)



np.shape(train_imgs[0])



model = Sequential()
model.add(Conv2D(filters=50, kernel_size=(3, 3), input_shape=(32, 32, 3), activation="relu"))

model.add(Dropout(0.5))

model.add(Conv2D(30, kernel_size=(3, 3), activation="relu"))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(54, activation="relu"))

model.add(Dense(num_classes, activation="softmax"))





model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer="adam",

              metrics=["accuracy"])

model.fit(train_imgs, out_y,

          batch_size=int(17500*0.8/100),

          epochs=4,

          validation_split = 0.2)
test_img_dir = "../input/test/test/"

test_img_pathes = [test_img_dir + fpath for fpath in sorted(os.listdir(test_img_dir))]

test_imgs = read_and_prep_images(test_img_pathes)



test_img_pathes[:5]
prediction = model.predict_proba(test_imgs)[:, 1]

answer = pd.DataFrame(columns=("id", "has_cactus"))



getFilename = lambda s: s.split("/")[-1]

for i in range(len(test_img_pathes)):

    #print(getFilename(test_img_pathes[i]), prediction[i])

    answer.loc[i] = (getFilename(test_img_pathes[i]), prediction[i])



answer.head()
answer.to_csv("submission.csv", index=False)