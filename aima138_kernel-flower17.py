import glob

import os

import numpy as np

import pandas as pd

from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array,array_to_img

import matplotlib.pyplot as plt

import tensorflow.keras as keras
# 0.参数初始化

image_height ,image_width,image_channels = 224,224,3

batch_size = 16

n_class = 17
# 1.获取所有图片路径

train_image_path  = "../input/17flowerclasses/17flowerclasses/train"  

test_image_path  = "../input/17flowerclasses/17flowerclasses/test"  

train_img_list = glob.glob(os.path.join(train_image_path,"*/*.jpg"))

test_img_list = glob.glob(os.path.join(test_image_path,"*/*.jpg"))

print(train_img_list[1],len(train_img_list))

print(test_img_list[1],len(test_img_list))
# 2.查看图片

image = load_img(train_img_list[0])

image = img_to_array(image, dtype=np.uint8)

plt.imshow(image)
# 3.数据集划分

# 训练集和验证集的扩展，包括旋转，裁剪，缩放，平移，翻转等。

trainGen = ImageDataGenerator(

    preprocessing_function=keras.applications.resnet50.preprocess_input,

    rotation_range=15,# 随机旋转角度30°

    shear_range=0.1, #裁剪比例

    zoom_range=0.1,#缩放比例

    width_shift_range=0.1, # 图片水平平移

    height_shift_range=0.1, # 图片垂直平移

    horizontal_flip=True,  # 允许水平垂直翻转

    vertical_flip=True, 

    validation_split=0.1#验证集比例

)



# 测试集数据处理

valid_Gen = ImageDataGenerator(

    preprocessing_function=keras.applications.resnet50.preprocess_input,

    validation_split=0.1

)



# 创建训练集数据

train_gen_data = trainGen.flow_from_directory(

    train_image_path, 

    target_size=(image_height ,image_width ), 

    batch_size=batch_size,# 分组数

    class_mode='categorical', 

    subset='training', 

    seed=0

)



# 创建验证集数据

validation_gen_data = valid_Gen.flow_from_directory(

    train_image_path,

    target_size=(image_height ,image_width ), 

    batch_size=batch_size,

    class_mode="categorical",

    subset="validation",

    seed=0 )



# 创建标签

labels = dict((v,k) for k,v in train_gen_data.class_indices.items())

print(labels)
# 4.构建resnet模型

weights_file_path = "../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

resnet50_model = keras.models.Sequential([

    keras.applications.ResNet50(

        include_top = False,

        pooling = 'avg',

        weights = weights_file_path,

    ),

    keras.layers.Dense(n_class,activation='softmax')

])

resnet50_model.layers[0].trainable = False

print(resnet50_model)
# 5.查看网络结构

resnet50_model.summary()
# 6.构建损失函数

resnet50_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 7.训练和保存模型

save_folder = 'save_cheackpoint'

if not os.path.exists(save_folder):

    os.makedirs(save_folder)

callbacks = [    keras.callbacks.ModelCheckpoint( os.path.join(save_folder,'resnet50_flowerclasses17_{epoch:02d}-{val_accuracy:.3f}.h5'),)   ]

history = resnet50_model.fit_generator(

            train_gen_data, steps_per_epoch=1071//batch_size,

            validation_data=validation_gen_data,validation_steps=119//batch_size,

            epochs=30,

            callbacks=callbacks,)
# 8.查看训练结果

history.history

def plot_learning_curves(history):

    pd.DataFrame(history.history).plot(figsize=(8,5))

    plt.grid(True)

    plt.gca().set_ylim(0,1)

    plt.show()



plot_learning_curves(history)
# 9.模型评估

x_test, y_test = validation_gen_data.__getitem__(1)

resnet50_model.evaluate(x_test,y_test)
# 10.保存模型

saveFolderName = "/kaggle/working/savemodel"

if not os.path.exists(saveFolderName):

    os.makedirs(saveFolderName)

file_name = 'flowerclass_resnet_model.h5'

filepath = os.path.join(saveFolderName,file_name)

resnet50_model.save(filepath)
# 11.保存训练过程中的准确率值

saveFolderName = "/kaggle/working/savecsv"

if not os.path.exists(saveFolderName):

    os.makedirs(saveFolderName)

file_name = 'df_resnet50_data.csv'

filepath = os.path.join(saveFolderName,file_name)

df = pd.DataFrame(history.history)

df.to_csv(filepath)