# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("./"))



# Any results you write to the current directory are saved as output.
class ImageDataGenerator_self(object):

    def __init__(self):

        self.reset()



    def reset(self):

        self.images = []

        self.labels = []



    def flow_from_directory(self, directory, classes, batch_size=32):

        # LabelEncode(classをint型に変換)するためのdict

        classes = {v: i for i, v in enumerate(sorted(classes))}

        while True:

            # ディレクトリから画像のパスを取り出す

            for path in pathlib.Path(directory).iterdir():

                # 画像を読み込みRGBへの変換、Numpyへの変換を行い、配列(self.iamges)に格納

                with Image.open(path) as f:

                    self.images.append(np.asarray(f.convert('RGB'), dtype=np.float32))

                # ファイル名からラベルを取り出し、配列(self.labels)に格納

                _, y = path.stem.split('_')

                self.labels.append(to_categorical(classes[y], len(classes)))



                # ここまでを繰り返し行い、batch_sizeの数だけ配列(self.iamges, self.labels)に格納

                # batch_sizeの数だけ格納されたら、戻り値として返し、配列(self.iamges, self.labels)を空にする

                if len(self.images) == batch_size:

                    inputs = np.asarray(self.images, dtype=np.float32)

                    targets = np.asarray(self.labels, dtype=np.float32)

                    self.reset()

                    yield inputs, targets



    def flow_from_dir2(self, data_dir, data_list, label_train, classes, batch_size=32):

        label_train = pd.read_csv(label_csv,index_col=0)

        while True:

            for img_path in data_list:

                img = cv2.imread(data_dir + "/" + img_path)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

                #img = img.transpose((2, 0, 1))

                #img = img.reshape((1,) + img.shape)

                self.images.append(img)



                # ファイル名からラベルを取り出し、配列(self.labels)に格納

                #_, y = path.stem.split('_')

                y = label_train.loc[img_path,"has_cactus"]



                self.labels.append(keras.utils.to_categorical(classes[y], len(classes)))



                # ここまでを繰り返し行い、batch_sizeの数だけ配列(self.iamges, self.labels)に格納

                # batch_sizeの数だけ格納されたら、戻り値として返し、配列(self.iamges, self.labels)を空にする

                if len(self.images) == batch_size:

                    inputs = np.asarray(self.images, dtype=np.float32)

                    targets = np.asarray(self.labels, dtype=np.float32)

                    self.reset()

                    yield inputs, targets
import csv,os

import numpy as np

import pandas as pd

import keras

train_order = os.listdir("../input/train/train")

pd_train_order = pd.DataFrame(train_order,columns = ["id"])

label_train = pd.read_csv("../input/train.csv")

df_train_label = pd.merge(pd_train_order, label_train)
import random

data_dir="../input/train/train"

data_list=os.listdir(data_dir)

label_csv="../input/train.csv"

classes=["0","1"]

batch_size = 10

val_ratio = 0.1



train_list = []

val_list = []



for data in data_list:

    if random.random() > val_ratio:

        train_list.append(data)

    else:

        val_list.append(data)

print(len(train_list))

print(len(val_list))



train_datagen=ImageDataGenerator_self()

test_datagen=ImageDataGenerator_self()

from matplotlib import pyplot as plt

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.layers import Input, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.utils import plot_model

from keras.callbacks import ModelCheckpoint

import cv2



img_height,img_width,img_channel = cv2.imread(data_dir + "/" + data_list[0]).shape

base_model = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=(img_width,img_height,img_channel))



n_categories = 2



x=base_model.output

x=GlobalAveragePooling2D()(x)

x=Dense(1024,activation='relu')(x)

prediction=Dense(n_categories,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=prediction)



model.compile(optimizer=optimizers.SGD(lr=0.0001,momentum=0.9),

              loss='categorical_crossentropy',

              metrics=['accuracy'])



#model.summary()
os.mkdir("/output")

fpath = '/output/weights.{epoch:03d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'

modelCheckpoint = ModelCheckpoint(filepath = fpath,

                                  monitor='loss',

                                  verbose=1,

                                  save_best_only=True,

                                  save_weights_only=False,

                                  mode='min',

                                  period=1)



model.fit_generator(

    generator=train_datagen.flow_from_dir2(data_dir=data_dir, data_list=train_list, label_train=label_train, classes=classes,batch_size=batch_size),

    #generator=train_datagen.flow_from_dir2(data_dir, data_list, label_csv,classes,batchsize),

    steps_per_epoch=int(len(data_list) / batch_size),

    epochs=100,

    verbose=2,

    validation_data=test_datagen.flow_from_dir2(data_dir=data_dir, data_list=val_list, label_train=label_train, classes=classes,batch_size=batch_size),

    validation_steps=int(len(val_list) / batch_size),

    callbacks=[modelCheckpoint]

    )
model_json_str = model.to_json()

open("/output/vgg16.json","w").write(model_json_str)
import pathlib

from keras.models import model_from_json

weight_file = os.listdir("/output")

latest_time = 0.0

for weight in weight_file:

    if ".hdf5" in weight:

        st = pathlib.Path("/output/" + weight).stat()

        if latest_time < st.st_mtime:

            latest_time = st.st_mtime

            latest_weight = weight

print(latest_weight)

print(latest_time)



model = model_from_json(open("/output/" + "/vgg16.json").read())

model.load_weights("/output/" + latest_weight)

model.summary()
output = [["id","has_cactus"]]

pred_list = os.listdir("../input/test/test")

for pred_img in pred_list:

    img = cv2.cvtColor(cv2.imread("../input/test/test/" + pred_img), cv2.COLOR_BGR2RGB).astype(np.float32)

    img = img.reshape((1,) + img.shape)

    result = model.predict(img)

    output.append([os.path.basename(pred_img),result[0][1]])



with open('/output/pred_result.csv', 'w') as f:

    writer = csv.writer(f)

    writer.writerows(output)
sub = pd.read_csv('/output/pred_result.csv')

sub.to_csv("submission.csv",index=False)
out = os.listdir("/")

print(out)