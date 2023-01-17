from tqdm import tqdm

import os

import numpy as np

from keras.preprocessing.image import img_to_array, load_img



class Fer2013(object):

    def __init__(self):

        """

        构造函数

        """

#       数据集文件夹

        self.folder = '/kaggle/input/expressionrecognition'



    def gen_train(self):

        """

        产生训练数据

        :return expressions:读取文件的顺序即标签的下标对应

        :return x_train: 训练数据集

        :return y_train： 训练标签

        """

        folder = os.path.join(self.folder, 'Training')

        # 各个类别

        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'normal']

        x_train = []

        y_train = []

        for i in tqdm(range(len(expressions))):

            expression_folder = os.path.join(folder, expressions[i])

            images = os.listdir(expression_folder)

            for j in range(len(images)):

#               读取训练集的图片

                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")

                img = img_to_array(img)  

                x_train.append(img)

#               将训练集打上标签anger:0 disgust:1等等

                y_train.append(i)

        x_train = np.array(x_train).astype('float32') / 255.

        y_train = np.array(y_train).astype('int')

        return expressions, x_train, y_train



    def gen_train_no(self):

        """

        另一种方法获得训练数据与上面的是差不多的效果

        :return expressions:读取文件的顺序即标签的下标对应

        :return x_train: 训练数据集

        :return y_train： 训练标签

        """

        folder = os.path.join(self.folder, 'Training')

        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'normal']

        x_train = []

        y_train = []

        import cv2

        for i in tqdm(range(len(expressions))):

            expression_folder = os.path.join(folder, expressions[i])

            images = os.listdir(expression_folder)

            for j in range(len(images)):

                img = cv2.imread(os.path.join(expression_folder, images[j]), cv2.IMREAD_GRAYSCALE)

                x_train.append(img)

                y_train.append(i)

        x_train = np.array(x_train)

        y_train = np.array(y_train).astype('int')

        return expressions, x_train, y_train



    def gen_valid(self):

        """

        产生验证集数据，与训练集一样操作

        :return:

        """

        folder = os.path.join(self.folder, 'PublicTest')

        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'normal']

        x_valid = []

        y_valid = []

        for i in tqdm(range(len(expressions))):

            expression_folder = os.path.join(folder, expressions[i])

            images = os.listdir(expression_folder)

            for j in range(len(images)):

                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")

                img = img_to_array(img)  

                x_valid.append(img)

                y_valid.append(i)

        x_valid = np.array(x_valid).astype('float32') / 255.

        y_valid = np.array(y_valid).astype('int')

        return expressions, x_valid, y_valid



    def gen_valid_no(self):

        """

        这里也是另一种方法获得验证数据

        :return expressions:读取文件的顺序即标签的下标对应

        :return x_train: 验证数据集

        :return y_train： 验证标签

        """

        folder = os.path.join(self.folder, 'PublicTest')

        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'normal']

        x_train = []

        y_train = []

        import cv2

        for i in tqdm(range(len(expressions))):

            expression_folder = os.path.join(folder, expressions[i])

            images = os.listdir(expression_folder)

            for j in range(len(images)):

                img = cv2.imread(os.path.join(expression_folder, images[j]), cv2.IMREAD_GRAYSCALE)

                x_train.append(img)

                y_train.append(i)

        x_train = np.array(x_train)

        y_train = np.array(y_train).astype('int')

        return expressions, x_train, y_train



    def gen_test(self):

        """

        产生测试集数据

        :return:

        """

        folder = os.path.join(self.folder, 'PrivateTest')

        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'normal']

        x_test = []

        y_test = []

        for i in tqdm(range(len(expressions))):

            expression_folder = os.path.join(folder, expressions[i])

            images = os.listdir(expression_folder)

            for j in range(len(images)):

                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")

                img = img_to_array(img)  # 灰度化

                x_test.append(img)

                y_test.append(i)

        x_test = np.array(x_test).astype('float32') / 255.

        y_test = np.array(y_test).astype('int')

        return expressions, x_test, y_test



    def gen_test_no(self):

        """

        这里也是另一种方法获得测试数据

        :return expressions:读取文件的顺序即标签的下标对应

        :return x_train: 测试数据集

        :return y_train： 测试标签

        """

        folder = os.path.join(self.folder, 'PrivateTest')

        # 这里原来是list出多个表情类别的文件夹，后来发现服务器linux顺序不一致，会造成问题，所以固定读取顺序

        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'normal']

        x_train = []

        y_train = []

        import cv2

        for i in tqdm(range(len(expressions))):

            expression_folder = os.path.join(folder, expressions[i])

            images = os.listdir(expression_folder)

            for j in range(len(images)):

                img = cv2.imread(os.path.join(expression_folder, images[j]), cv2.IMREAD_GRAYSCALE)

                x_train.append(img)

                y_train.append(i)

        x_train = np.array(x_train)

        y_train = np.array(y_train).astype('int')

        return expressions, x_train, y_train
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, AveragePooling2D

from keras.models import Model

from keras.layers.advanced_activations import PReLU



# 这里分别是三个模型，可以分别跑一下，我只跑了第三个一个，可以将这三个模型进行对比比较，最后写进论文里。

# 里面的每层建议百度一下，仔细看看。

def CNN1(input_shape=(48, 48, 1), n_classes=7):

    """

    参考VGG思路设计的第一个模型

    :param input_shape: 输入图片的尺寸

    :param n_classes: 目标类别数目

    :return:

    """

    # input

    input_layer = Input(shape=input_shape)

    # block1

    x = Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input_layer)

    x = Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Dropout(0.5)(x)

    # block2

    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)

    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Dropout(0.5)(x)

    # block3

    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Dropout(0.5)(x)

    # fc

    x = Flatten()(x)

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)

    output_layer = Dense(n_classes, activation='softmax')(x)



    model = Model(inputs=input_layer, outputs=output_layer)

    return model





def CNN2(input_shape=(48, 48, 1), n_classes=7):

    """

    参考论文Going deeper with convolutions在输入层后加一层的1*1卷积增加非线性表示

    :param input_shape:

    :param n_classes:

    :return:

    """

    # input

    input_layer = Input(shape=input_shape)

    # block1

    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)

    x = Conv2D(32, (5, 5), strides=1, padding='same', activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # block2

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # block3

    x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # fc

    x = Flatten()(x)

    x = Dense(2048, activation='relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(n_classes, activation='softmax')(x)



    model = Model(inputs=input_layer, outputs=x)

    return model





def CNN3(input_shape=(48, 48, 1), n_classes=7):

    """

    参考论文A Compact Deep Learning Model for Robust Facial Expression Recognition实现

    :param input_shape:

    :param n_classes:

    :return:

    """

    # input

    input_layer = Input(shape=input_shape)

    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)

    # block1

    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)

    x = PReLU()(x)

    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)

    x = PReLU()(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # block2

    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)

    x = PReLU()(x)

    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)

    x = PReLU()(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # fc

    x = Flatten()(x)

    x = Dense(2048, activation='relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(n_classes, activation='softmax')(x)



    model = Model(inputs=input_layer, outputs=x)

    return model
from keras.utils import to_categorical



# 获取到训练集、测试集的数据

expressions, x_train, y_train = Fer2013().gen_train()

expressions, x_valid, y_valid = Fer2013().gen_valid()



# target编码，使得y变为one-hot编码，也就是我们说的独热码，可以自己了解一下。

import numpy as np

y_train = to_categorical(y_train).reshape(y_train.shape[0], -1)

y_valid = to_categorical(y_valid).reshape(y_valid.shape[0], -1)

print(y_train.shape)

print(y_valid.shape)
model = CNN3(input_shape=(48, 48, 1), n_classes=7)



from keras.callbacks import ModelCheckpoint

from keras.optimizers import SGD



# 使用随机梯度下降的方法来作为优化器，也可以了解一下其他的优化器，尝试使用其他的优化器来跑咱们这个模型，写进论文里。

# 很多参数也都是可以调节的，可以自己调节一下试试，有时候调节一下参数会带来很大的提升。

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# 使用交叉熵作为损失函数，也就是loss，分类大多用这一损失函数，你也可以了解一下其他的损失函数，写进论文里。

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# 模型存储

callback = [

    ModelCheckpoint('/kaggle/working/models/cnn3_best_weights.h5', monitor='val_acc', verbose=True, save_best_only=True, save_weights_only=True)]

epochs = 100

batch_size = 128



from keras.preprocessing.image import ImageDataGenerator

# 使用ImageDataGenerator进行data augmentation，也就是数据增强，以增强模型的泛化能力。可以了解一下数据增强，写进论文里。

# 里面的参数自己百度一哈哈。

train_generator = ImageDataGenerator(rotation_range=10, 

                                     width_shift_range=0.05, 

                                     height_shift_range=0.05, 

                                     horizontal_flip=True, 

                                     shear_range=0.2, 

                                     zoom_range=0.2).flow(x_train, y_train, batch_size=batch_size)

valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=batch_size)



# 进行训练

history_fer2013 = model.fit_generator(train_generator, 

                              steps_per_epoch=len(y_train)//batch_size, 

                              epochs=epochs, 

                              validation_data=valid_generator, 

                              validation_steps=len(y_valid)//batch_size, 

                              callbacks=callback)
# 使用我们最开始划分的还剩的测试集进行测试

_, x_test, y_test = Fer2013().gen_test()

pred = model.predict(x_test)

pred = np.argmax(pred, axis=1)

print(np.sum(pred.reshape(-1) == y_test.reshape(-1)) / y_test.shape[0])
# 绘制图像

import matplotlib.pyplot as plt



# 损失

plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)

plt.plot(np.arange(len(history_fer2013.history['loss'])), history_fer2013.history['loss'], label='fer2013 train loss')



plt.plot(np.arange(len(history_fer2013.history['val_loss'])), history_fer2013.history['val_loss'], label='fer2013 valid loss')

plt.legend(loc='best')

# 准确率

plt.subplot(1, 2, 2)

plt.plot(np.arange(len(history_fer2013.history['accuracy'])), history_fer2013.history['accuracy'], label='fer2013 train accuracy')



plt.plot(np.arange(len(history_fer2013.history['val_accuracy'])), history_fer2013.history['val_accuracy'], label='fer2013 valid accuracy')

plt.legend(loc='best')

plt.savefig('/kaggle/working/loss.png')

plt.show()