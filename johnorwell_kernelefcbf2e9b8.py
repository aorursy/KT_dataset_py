import os

import shutil



# 数据集解压之后的目录

original_dataset_dir = '/data/nextcloud/dbc2017/files/jupyter/train'

# 存放小数据集的目录

base_dir = '/data/nextcloud/dbc2017/files/jupyter//cats_and_dogs_small'

os.mkdir(base_dir)
# 建立训练集、验证集、测试集目录

train_dir = os.path.join(base_dir, 'train')

os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')

os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')

os.mkdir(test_dir)



# 将猫狗照片按照训练、验证、测试分类

train_cats_dir = os.path.join(train_dir, 'cats')

os.mkdir(train_cats_dir)



train_dogs_dir = os.path.join(train_dir, 'dogs')

os.mkdir(train_dogs_dir)



validation_cats_dir = os.path.join(validation_dir, 'cats')

os.mkdir(validation_cats_dir)



validation_dogs_dir = os.path.join(validation_dir, 'dogs')

os.mkdir(validation_dogs_dir)



test_cats_dir = os.path.join(test_dir, 'cats')

os.mkdir(test_cats_dir)



test_dogs_dir = os.path.join(test_dir, 'dogs')

os.mkdir(test_dogs_dir)
# 切割数据集

# 训练集:验证集:测试集

# 3:1:1

# 猫图片训练集

fnames = ['cat.{}.jpg'.format(i) for i in range(3000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dat = os.path.join(train_cats_dir, fname)

    shutil.copyfile(src, dat)

# 猫图片验证集 验证集用于“训练”模型的超参数

fnames = ['cat.{}.jpg'.format(i) for i in range(3000, 4000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dat = os.path.join(validation_cats_dir, fname)

    shutil.copyfile(src, dat)

# 猫图片测试集 测试集用于估计模型对样本的泛化误差

fnames = ['cat.{}.jpg'.format(i) for i in range(4000, 5000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dat = os.path.join(test_cats_dir, fname)

    shutil.copyfile(src, dat)

# 狗图片训练集

fnames = ['dog.{}.jpg'.format(i) for i in range(3000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dat = os.path.join(train_dogs_dir, fname)

    shutil.copyfile(src, dat)

# 狗图片验证集

fnames = ['dog.{}.jpg'.format(i) for i in range(3000, 4000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dat = os.path.join(validation_dogs_dir, fname)

    shutil.copyfile(src, dat)



fnames = ['dog.{}.jpg'.format(i) for i in range(4000, 5000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dat = os.path.join(test_dogs_dir, fname)

    shutil.copyfile(src, dat)
!nvidia-smi
from keras import layers

from keras import models

import matplotlib.pyplot as plt

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator



base_dir = '/data/nextcloud/dbc2017/files/jupyter//cats_and_dogs_small'



train_dir = base_dir+'/train'

validation_dir = base_dir+'/validation'
# 简单版cnn网络模型

model = models.Sequential()

#   CBAPD

# 32个卷积核 卷积和尺寸为3×3 激活函数为relu 输入图片尺寸150×150 3通道

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

# 池化层卷积尺寸为2×2

model.add(layers.MaxPool2D((2, 2)))



model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPool2D((2, 2)))



model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPool2D((2, 2)))



model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPool2D((2, 2)))



# 拉直层

model.add(layers.Flatten())

# 丢弃层

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])



# 调整像素值 将0~255区间的像素值减少到0~1区间中，Cnn更喜欢处理小的输入值

train_datagen = ImageDataGenerator(rescale=1. / 255, )

test_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(

    directory=train_dir,

    target_size=(150, 150),

    batch_size=20,

    class_mode='binary'

)



validation_generator = test_datagen.flow_from_directory(

    directory=validation_dir,

    target_size=(150, 150),

    batch_size=20,

    class_mode='binary'

)

# 使用fit_genertor在模型中填充数据

history = model.fit_generator(

    train_generator,

    steps_per_epoch=100,

    epochs=30,

    validation_data=validation_generator,

    validation_steps=50

)

# 保存模型

model.save('cats_and_dogs_small_0.h5')
model.load_weights('cats_and_dogs_small_0.h5')



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

# 显示acc曲线，loss曲线

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
from keras import layers

from keras import models

import matplotlib.pyplot as plt

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator



base_dir = '/data/nextcloud/dbc2017/files/jupyter//cats_and_dogs_small'



train_dir = base_dir+'/train'

validation_dir = base_dir+'/validation'

# train_dir = r'/kaggle/working/cats_and_dogs_small/train'

# validation_dir = r'/kaggle/working/cats_and_dogs_small/validation'base_dir
# 优化版本data augmentation

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

model.add(layers.MaxPool2D((2, 2)))



model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPool2D((2, 2)))



model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPool2D((2, 2)))



model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPool2D((2, 2)))



model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])







# 调整像素值

# 使用数据增强优化

train_datagen = ImageDataGenerator(

    rescale=1./255, 

    rotation_range=40, 

    width_shift_range=0.2, 

    height_shift_range=0.2, 

    shear_range=0.2, 

    zoom_range=0.2,

    horizontal_flip=True

)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

    directory=train_dir,

    target_size=(150, 150),

    batch_size=32,

    class_mode='binary'

)



validation_generator = test_datagen.flow_from_directory(

    directory=validation_dir,

    target_size=(150, 150),

    batch_size=32, # 调大

    class_mode='binary'

)



# 使用实时数据增益的批数据对模型进行拟合

history = model.fit_generator(

    train_generator,

    steps_per_epoch=100,

    epochs=100, # 调大

    validation_data=validation_generator,

    validation_steps=50

)



model.save('cats_and_dogs_small_1.h5')

# 1 hour
# 这个版本的模型需要训练一个小时 可以直接加载模型参数

model.load_weights('cats_and_dogs_small_1.h5')



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
from keras.applications import VGG16

import os

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from keras import models

from keras import layers

from keras import optimizers

import matplotlib.pyplot as plt



# Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5

# vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5

# 放在 .keras/models

# 不增加数据量 使用预训练网络

conv_base = VGG16(weights='imagenet',

                  include_top=False, #是否包含全连接分类器 显然在ImageNet有上千分类在这里，我们不需要的

                  input_shape=(150, 150, 3))



base_dir = '/data/nextcloud/dbc2017/files/jupyter//cats_and_dogs_small'



train_dir = base_dir+'/train'

validation_dir = base_dir+'/validation'

test_dir=base_dir+'/test'



# base_dir = '/kaggle/working/cats_and_dogs_small'

# train_dir = os.path.join(base_dir, 'train')

# validation_dir = os.path.join(base_dir, 'validation')

# test_dir = os.path.join(base_dir, 'test')
datagen = ImageDataGenerator(rescale=1./255)

batch_size = 20



# 从预训练的baseline卷积层中提取特征

def extarct_features(directory, sample_count):

    features = np.zeros(shape=(sample_count, 4, 4, 512))

    labels = np.zeros(shape=(sample_count))

    generator = datagen.flow_from_directory(

        directory,

        target_size=(150, 150),

        batch_size=batch_size,

        class_mode='binary'

    )



    i = 0

    for inputs_batch, labels_batch in generator:

        features_batch = conv_base.predict(inputs_batch)

        features[i * batch_size : (i + 1) * batch_size] = features_batch

        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        i += 1

        if i * batch_size >= sample_count:

            break



    return features, labels





train_features, train_labels = extarct_features(train_dir, 2000)

validation_features, validation_labels = extarct_features(validation_dir, 1000)

test_features, test_labels = extarct_features(test_dir, 1000)



train_features = np.reshape(train_features, (2000, 4 * 4 * 512))

validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))

test_features = np.reshape(test_features, (1000, 4 * 4 * 512))



model = models.Sequential()

model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(optimizer=optimizers.RMSprop(lr=2e-5),

              loss='binary_crossentropy',

              metrics=['acc'])



history = model.fit(train_features, train_labels,

                    epochs=30,

                    batch_size=20,

                    validation_data=(validation_features, validation_labels))

model.save('cats_and_dogs_small_2.h5')

# epoch=30 

# 1 min
model.load_weights('cats_and_dogs_small_2.h5')



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()





# 验证集准确率已经达到90%，要好于之前，一贯在小的数据集上做训练。

#

# 但仍然有过拟合的问题，很可能是因为我们并没有用到数据增强

#

# 下一个版本中我们将加上数据增强
from keras.applications import VGG16

from keras.preprocessing.image import ImageDataGenerator

from keras import models

from keras import layers

from keras import optimizers

import matplotlib.pyplot as plt



base_dir = '/data/nextcloud/dbc2017/files/jupyter//cats_and_dogs_small'



train_dir = base_dir+'/train'

validation_dir = base_dir+'/validation'

test_dir=base_dir+'/test'

# train_dir = r'/kaggle/working/cats_and_dogs_small/train'

# validation_dir = r'/kaggle/working/cats_and_dogs_small/validation'
conv_base = VGG16(weights='imagenet',

                  include_top=False,

                  input_shape=(150, 150, 3))

conv_base.trainable = False





model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))

model.add(layers.Dense(1, activation='sigmoid'))



# 增加了数据增强

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)





test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

    directory=train_dir,

    target_size=(150, 150),

    batch_size=32,

    class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

    directory=validation_dir,

    target_size=(150, 150),

    batch_size=32,

    class_mode='binary')



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=2e-5),

              metrics=['acc'])



history = model.fit_generator(

    train_generator,

    steps_per_epoch=100,

    epochs=100,

    validation_data=validation_generator,

    validation_steps=50)



model.save('cats_and_dogs_small_3.h5')

# epoch=100

# 1 hour
# 这个版本的模型需要训练一个小时 可以直接加载模型参数

model.load_weights('cats_and_dogs_small_3.h5')



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()



# 增大数据集

# 有效的解决了过拟合的问题
from keras.applications import VGG16

from keras.preprocessing.image import ImageDataGenerator

from keras import models

from keras import layers

from keras import optimizers

import matplotlib.pyplot as plt



base_dir = '/data/nextcloud/dbc2017/files/jupyter//cats_and_dogs_small'



train_dir = base_dir+'/train'

validation_dir = base_dir+'/validation'

test_dir=base_dir+'/test'



# train_dir = r'/kaggle/working/cats_and_dogs_small/train'

# validation_dir = r'/kaggle/working/cats_and_dogs_small/validation'

# test_dir = r'/kaggle/working/cats_and_dogs_small/test'
conv_base = VGG16(weights='imagenet',

                  include_top=False,

                  input_shape=(150, 150, 3))

set_trainable = False



# 微调

# 解冻之前固定的vgg16模型

# 1在一个已经训练好的baseline网络上添加自定义网络

# 2冻结baseline网络

# 3训练我们所添加的部分

# 4解冻一些baseline网络中的卷积层

# 5将我们所添加的部分与解冻的卷积层相连接

for layer in conv_base.layers:

    if layer.name == 'block5_conv1':

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False





model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))

model.add(layers.Dense(1, activation='sigmoid'))



train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')





test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

    directory=train_dir,

    target_size=(150, 150),

    batch_size=32,

    class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

    directory=validation_dir,

    target_size=(150, 150),

    batch_size=32,

    class_mode='binary')



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-5),

              metrics=['acc'])



history = model.fit_generator(

    train_generator,

    steps_per_epoch=100,

    epochs=100,

    validation_data=validation_generator,

    validation_steps=50

)

model.save('cats_and_dogs_small_4.h5')

# epoch=100

# 1 hour
# 这个版本的模型需要训练一个小时 可以直接加载模型参数

model.load_weights('cats_and_dogs_small_4.h5')





def smooth_curve(points, factor=0.8):

    smoothed_points = []

    for point in points:

        if smoothed_points:

            previous = smoothed_points[-1]

            smoothed_points.append(int(previous * factor + point * (1 - factor)))

        else:

            smoothed_points.append(point)





acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')

# plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed validation acc')

# plt.title('Training and validation accuracy')

# plt.legend()



epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Smoothed training acc')

plt.plot(epochs, val_acc, 'b', label='Smoothed validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



# plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')

# plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')

# plt.title('Training and validation loss')

# plt.legend()



plt.plot(epochs, loss, 'bo', label='Smoothed training loss')

plt.plot(epochs, val_loss, 'b', label='Smoothed validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()



test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size=(150, 150),

    batch_size=20,

    class_mode='binary'

)

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)

print('test acc:', test_acc)