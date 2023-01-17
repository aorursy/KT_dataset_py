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
import numpy as np

import pandas as pd 

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random

import os

print(os.listdir('../input/dogs-vs-cats'))
print(os.listdir('./'))
!unzip ../input/dogs-vs-cats/train.zip
print(os.listdir('./'))
!unzip ../input/dogs-vs-cats/test1.zip
print(os.listdir('./train'))
## 将猫的图片放到一个train/cats 文件夹下。

## 将狗的图片放到一个train/dogs 文件夹下。



root_path = './train'

train_cats_path = os.path.join(root_path,'cats')

train_dogs_path = os.path.join(root_path,'dogs')



if not os.path.exists(train_cats_path):

    os.mkdir(train_cats_path)

if not os.path.exists(train_dogs_path):

    os.mkdir(train_dogs_path)
import shutil  #用于拷贝文件。

import glob # 用于列出特定文件地址。



list_cat_path = glob.glob(root_path+'/cat.*.jpg')

list_dog_path = glob.glob(root_path+'/dog.*.jpg')

len(list_cat_path)

len(list_dog_path)
## 总共12500 张猫和狗的图像。

# 然后将他们分别拷贝到特定文件夹小。

for cat_path in list_cat_path:

    src = cat_path

    dst = train_cats_path

    

    shutil.move(src,dst)
for dog_path in list_dog_path:

    src = dog_path

    dst = train_dogs_path

#     if not os.path.exists(dst):

    shutil.move(src,dst)
## 对测试集进行同样的操作。

root_test_path = './test1'

test_cats_path = os.path.join(root_test_path,'cats')

test_dogs_path = os.path.join(root_test_path,'dogs')

if not os.path.exists(test_cats_path):

    os.mkdir(test_cats_path)

if not os.path.exists(test_dogs_path):

    os.mkdir(test_dogs_path)
# 进行移动

list_path_test_dogs = glob.glob(root_test_path+'/dog.*.jpg')

list_path_test_cats = glob.glob(root_test_path+'/cat.*.jpg')



for dog_path in list_path_test_dogs:

    src = dog_path

    dst = test_dogs_path

    shutil.move(src,dst)

for cat_path in list_path_test_cats:

    src = cat_path

    dst = test_cats_path

    shutil.move(src,dst)
## 验证集。

val_root = './val'

if not os.path.exists(val_root):

    os.mkdir(val_root)

path_val_cats = os.path.join(val_root,'cats')

path_val_dogs = os.path.join(val_root,'dogs')



if not os.path.exists(path_val_cats):

    os.mkdir(path_val_cats)

if not os.path.exists(path_val_dogs):

    os.mkdir(path_val_dogs)

list_cat_path[-2:]
len(list_dog_path)
# 一部分训练集的图片移动到验证集中。

# 移动2500张图片分别到猫狗文件夹中。list_cat_path的最后2500张，和list_dog_path的最后2500张。

# train_cats_path and train_dogs_path文件夹下的最后2500张图片。



list1 = glob.glob(train_cats_path+'/cat.*.jpg')

list2 = glob.glob(train_dogs_path+'/dog.*.jpg')





for path in list1[-2500:]:

    src = path

    dst = path_val_cats

    

    shutil.move(src,dst)

    

for path in list2[-2500:]:

    src = path

    dst = path_val_dogs

   

    shutil.move(src,dst)

x =os.listdir(path_val_cats)

len(x)
## 运用keras 的imagegeneator来生成训练数据。

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1.0/255,

                                  rotation_range=40,

                                  shear_range=0.2,

                                  width_shift_range=0.2,

                                  height_shift_range=0.2,

                                  zoom_range=0.2,

                                  horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0/255)



# 从文件夹生成数据。



train_generator = train_datagen.flow_from_directory(root_path,

                                                   target_size=(150,150),

                                                   batch_size=20,

                                                   class_mode='binary')



test_generator = test_datagen.flow_from_directory(val_root,

                                                 target_size=(150,150),

                                                 batch_size=20,

                                                 class_mode='binary')



from keras import models

from keras import layers

#### 用vgg16 来进行训练。

from keras.applications import VGG16,ResNet50V2

conv_base = ResNet50V2(weights='imagenet',

                  include_top=False,

                  input_shape=(150,150,3))



conv_base.summary()
# conv_base.summary()

from keras import metrics,optimizers

model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

# model.add(layers.Dropout(0.4))  #天机dropout层。

model.add(layers.Dense(1024,activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1,activation='sigmoid'))



# print(model.summary())



conv_base.trainable = True



set_trainable = False

for layer in conv_base.layers:  # balock5 后边的所有层都可训练。

    if layer.name == 'conv5_block1_1_conv':   # 这里可以根据前边的baselayer 打印的出的层的名字，来选择冻结层。

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False

        

model.compile(loss='binary_crossentropy',

              optimizer = optimizers.RMSprop(lr=1e-5),

              metrics=['acc'])

history = model.fit_generator(train_generator,

                              steps_per_epoch=100,

                              epochs=50,

                              validation_data=test_generator,

                              validation_steps=50)



model.save('./first_cat_dogmod.h5')
## 画出损失和准确率的图形。



#### 使曲线变得平滑。

# 应用移动平均。

def smooth_curve(points, factor=0.8):

    smoot_points = []

    for point in points:

        if smoot_points:

            prev_point = smoot_points[-1]

            temp_point = prev_point*factor + point*(1-factor)

            smoot_points.append(temp_point)

        else:

            smoot_points.append(point)

    return smoot_points

# 画出更加平滑的曲线。

#  训练过程的损失函数和准确率。。

import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1,len(acc)+1)

plt.plot(epochs,smooth_curve(acc),'b--',label='training acc')

plt.plot(epochs,smooth_curve(val_acc),'b',label='validation acc')

plt.title('trianing and validation accuracy')

plt.legend()

plt.grid()

plt.figure()

plt.plot(epochs,smooth_curve(loss),'b--',label='trainning loss')

plt.plot(epochs,smooth_curve(val_loss),'b',label='validation loss')

plt.title('trainning and validation loss')

plt.grid()

plt.legend()

plt.show()
import cv2

from keras.preprocessing import image

def prepare_data(list_of_images):

    """

    Returns two arrays: 

        x is an array of resized images

        y is an array of labels

    """

    x = [] # images as arrays

    y = [] # labels

    

    for imagex in list_of_images:

#         x.append(cv2.resize(cv2.imread(image), (img_width=150,img_height=150), interpolation=cv2.INTER_CUBIC))

        img = image.load_img(imagex, target_size=(150,150))

        imgx = image.img_to_array(img)

        x.append(imgx)

    for i in list_of_images:

        if 'dog' in i:

            y.append(1)

        elif 'cat' in i:

            y.append(0)

        #else:

            #print('neither cat nor dog name present in images')

            

    return x, y
# 这里对测试集进行预测，然后提交结果。

# test_images_dogs_cats = os.listdir('./test1')

test_images_dogs_cats = glob.glob('./test1'+'/*.jpg')





X_test, Y_test = prepare_data(test_images_dogs_cats) #Y_test in this case will be []



test_datagen = ImageDataGenerator(rescale=1. / 255)



test_generator = test_datagen.flow(np.asarray(X_test), batch_size=1)

prediction_probabilities = model.predict_generator(test_generator, verbose=1)





counter = range(1, len(test_images_dogs_cats) + 1)

solution = pd.DataFrame({"id": counter, "label":list(prediction_probabilities)})

cols = ['label']



for col in cols:

    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)



solution.to_csv("dogsVScats.csv", index = False)





os.listdir('./test1')