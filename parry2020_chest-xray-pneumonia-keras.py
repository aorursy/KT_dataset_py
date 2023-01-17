import numpy as np

from tensorflow.python import keras

from keras import backend as K

from keras import layers

from keras.models import Model, Sequential

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix

from mlxtend.plotting import plot_confusion_matrix

import matplotlib.pyplot as plt
#可以使用print(os.listdir(train_dir))读取目录下的文件

train_dir = r'../input/chest-xray-pneumonia/chest_xray/train'

test_dir = r'../input/chest-xray-pneumonia/chest_xray/test'



# 设置基本参数

image_size = 150

batch_size = 50

nb_classes = 2



#使用ImageDataGenerator类进行图片预处理,作为图片生成器

train_datagen = ImageDataGenerator(rescale = 1./255,     # 所有像素点转换到0-1

                              width_shift_range = 0.1,   # 数据提升时图片水平偏移的幅度

                              height_shift_range = 0.1,  # 数据提升时图片竖直偏移的幅度

                              shear_range = 0.2,         # 设置剪切强度

                              horizontal_flip = True,    # 随机选择一半图片水平翻转

                              fill_mode ='nearest')      # 超出边界的点将根据本参数给定的方法进行处理,一般给定的有有‘constant’，‘nearest’，‘reflect’或‘wrap。



test_datagen = ImageDataGenerator(rescale = 1./255)



print("traning set: ")



# 通过实时数据增强生成张量图像数据批次。数据将不断循环（按批次）。

train_datagen = train_datagen.flow_from_directory(train_dir,              # 目标目录的路径。每个类应该包含一个子目录。任何在子目录树下的 PNG, JPG, BMP, PPM 或 TIF 图像，都将被包含在生成器中

                                               (image_size, image_size),  # 整数元组 (height, width)，默认：(256, 256)。所有的图像将被调整到的尺寸。

                                               batch_size=batch_size,     # 一批数据的大小（默认 32）

                                               class_mode='categorical')  # class_mode为categorical", "binary", "sparse"或None之一，决定了返回的标签的数组形式，categorical返回的是2D one-hot编码标签。



print("testing set: ")



test_datagen = test_datagen.flow_from_directory(test_dir,

                                             (image_size,image_size),

                                              batch_size=batch_size,

                                              class_mode='categorical')



#定义步数

train_steps = train_datagen.samples//batch_size # " // "表示整数除法

test_steps = test_datagen.samples//batch_size

# val_steps = val_datagen.samples//batch_size 如果不调用测试集，可以自己建立一个验证集
#为了增加程序和模型的兼容能力，加入判断，规定输入image_input的shape类型

if K.image_data_format()=='channels_first':  # 返回默认的图像的维度顺序（‘channels_last’或‘channels_first’）

   input_shape =(3,image_size,image_size)

else:

   input_shape =(image_size,image_size,3)   

   

# 输入特征是150x150x3的张量，其中150x150用于图像像素，3用于三个颜色通道

img_input = layers.Input(shape=input_shape)



# 第一个卷积层提取3x3x32（size of 2D convolution window:3x3，输出32维）的特征，使用线性整流函数（Rectified Linear Unit, ReLU），然后是具有2x2大小的最大池化层

x = layers.Conv2D(32,3,activation='relu')(img_input)

x = layers.MaxPooling2D(2)(x)



x = layers.Conv2D(64,3,activation='relu')(x)

x = layers.MaxPooling2D(2)(x)



x = layers.Conv2D(64,3,activation='relu')(x)

x = layers.MaxPooling2D(2)(x)



# 将特征图展平为一维数据（`1-dim`）张量，以便添加全连接层（dense）

x = layers.Flatten()(x)



# 使用`sigmoid`激活函数和128个神经元创建全连接层

x = layers.Dense(128,activation='sigmoid')(x)



# 按一定的概率随机断开输入神经云，防止过拟合

x = layers.Dropout(0.5)(x)                       



# 使用2个神经元和`softmax`激活函数创建输出层

output = layers.Dense(2,activation='softmax')(x)



model= Model(img_input,output)



model.summary()                                
# 给模型定义相应的参数

model.compile(loss='categorical_crossentropy', # 多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列

              optimizer =Adam(lr=0.0001),

              metrics = ['acc'] )              # 指标列表metrics：对分类问题，我们一般将该列表设置为metrics=['accuracy']



# 设置相应的训练迭代次数

epochs = 20



# 训练模型

history = model.fit_generator(train_datagen,

                             steps_per_epoch=train_steps,

                             epochs=epochs,

                             validation_data=test_datagen,

                             validation_steps=test_steps )      
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs_range = range(1, epochs + 1)



plt.figure(figsize=(8,4))

plt.plot(epochs_range, acc, label='Train Set')

plt.plot(epochs_range, val_acc, label='Test Set')

plt.legend(loc="best")

plt.xlabel('Epochs')

plt.title('Model Accuracy')

plt.show()



plt.figure(figsize=(8,4))

plt.plot(epochs_range, loss, label='Train Set')

plt.plot(epochs_range, val_loss, label='Test Set')

plt.legend(loc="best")

plt.xlabel('Epochs')

plt.title('Model Loss')

plt.show()
# 评估模型得到相应的准确率（Accuracy），精确率（Precision），召回率（recall）

Y_pred = model.predict_generator(test_datagen,test_steps+1)

y_pred = np.argmax(Y_pred,axis=1)



CM =confusion_matrix(test_datagen.classes,y_pred)

print("CM:",CM)



pneumonia_precision= CM[1][1] / (CM[1][0]+CM[1][1])

print("pnuemonia_precision:", pneumonia_precision)



pnuemonia_recall = CM[1][1] / (CM[1][1]+CM[0][1])

print('pnuemonia_recall:', pnuemonia_recall)



accuracy = (CM[0][0]+CM[1][1])/(CM[0][0]+CM[0][1]+CM[1][0]+CM[1][1])

print('accuracy:', accuracy)



target_names = ['Normal', 'Pneumonia'] 

print(classification_report(test_datagen.classes, y_pred, target_names=target_names))