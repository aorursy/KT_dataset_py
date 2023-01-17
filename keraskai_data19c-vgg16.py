import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import os



image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")





def list_images(basePath, contains=None):

    # return the set of files that are valid

    return list_files(basePath, validExts=image_types, contains=contains)





def list_files(basePath, validExts=None, contains=None):

    # loop over the directory structure

    for (rootDir, dirNames, filenames) in os.walk(basePath):

        # loop over the filenames in the current directory

        for filename in filenames:

            # if the contains string is not none and the filename does not contain

            # the supplied string, then ignore the file

            if contains is not None and filename.find(contains) == -1:

                continue



            # determine the file extension of the current file

            ext = filename[filename.rfind("."):].lower()



            # check to see if the file is an image and should be processed

            if validExts is None or ext.endswith(validExts):

                # construct the path to the image and yield it

                imagePath = os.path.join(rootDir, filename)

                yield imagePath
import random





imagePaths = sorted(list(list_images("/kaggle/input/data19c3/data19c")))  #list 是将元祖转换成列表  sorted进行排序

random.seed(42) #随机种子进行排序

random.shuffle(imagePaths) 
import cv2

from keras.utils import np_utils

import numpy as np





labels = []

data = []

for j,imagePath in enumerate(imagePaths): #获取图像

    image = cv2.imread(imagePath)



    image = cv2.resize(image, (224, 224))#设置图像的大小

    # cv2.imshow("ssdf",image)

    # cv2.waitKey(50)

    data.append(image) #extend 和 append 的区别  这里是每一个都加

    label = imagePath.split(os.path.sep)[-2]  #os.path.sep路径分割符号  #slit以分割符号进行分割 获取倒数第二个参数

    labels.append(label) #这里是获取列表

    

    aa = set(labels)

    num_classes  = len(aa)

data = np.array(data)

labels = np.array(labels)

labels = np_utils.to_categorical(labels, num_classes)
from sklearn.model_selection import train_test_split



(X_train, Y_train, X_valid, Y_valid) = train_test_split(data,

	labels, test_size=0.25, random_state=42)
print(X_train.shape)

print(Y_train.shape)

print(X_valid.shape)

print(Y_valid.shape)


from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dropout, Flatten, Dense

from keras import applications

from keras.models import Model

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, add,  Activation

from keras.optimizers import SGD

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, add, Flatten, Activation

from keras.layers.normalization import BatchNormalization

from keras.models import Model

import matplotlib.pyplot as plt

import numpy as np

from keras.utils import plot_model

from sklearn.metrics import log_loss



path = "/kaggle/input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

base_model = applications.VGG16(weights=path, include_top=False,

                                input_shape=(224, 224, 3)) 





for layer in base_model.layers[:15]: layer.trainable = False





top_model = Sequential()  # 自定义顶层网络

top_model.add(AveragePooling2D((7, 7), name='avg_pool'))

top_model.add(Flatten(input_shape=base_model.output_shape[1:]))  # 将预训练网络展平

top_model.add(Dense(256, activation='relu'))  # 全连接层，输入像素256

top_model.add(Dropout(0.5))  # Dropout概率0.5

top_model.add(Dense(13, activation='softmax'))  # 输出层，二分类



# top_model.load_weights("")  # 单独训练的自定义网络



model = Model(inputs=base_model.input, outputs=top_model(base_model.output))  # 新网络=预训练网络+自定义网络

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
img_rows, img_cols = 224, 224 # h,w

channel = 3

num_classes = 13

batch_size = 16 

nb_epoch = 500












H = model.fit(X_train, X_valid,

          batch_size=batch_size,

          epochs=nb_epoch,

          shuffle=True,

          verbose=1,

          validation_data=(Y_train, Y_valid),

          )





    

plt.style.use("ggplot")

plt.figure()

plt.ylim(0,3)

N = nb_epoch

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend(loc="upper left")

plt.savefig('result.png')