
# 导入本地数据集
import os, shutil

# 注意文件的位置！！！！！！！！！！！
base_dir = "/kaggle/input/dogsimages/dogImages"

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

train_Affenpinscher_dir = os.path.join(train_dir, '001.Affenpinscher')

print("have", len(os.listdir(train_Affenpinscher_dir)))
# 迁移学习 导入模型VGG19
from keras.applications import VGG19

conv_base = VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
conv_base.trainable = False
conv_base.summary()

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
 
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)
 
    
train_generator = train_datagen.flow_from_directory(
        train_dir,  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 224x224
        batch_size=32,
        class_mode='categorical') 
 
# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
# 数据增强
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

fnames = [os.path.join(train_Affenpinscher_dir, fname) for fname in os.listdir(train_Affenpinscher_dir)]

img_path = fnames[3]
img = load_img(img_path)  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
 
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(tf.keras.preprocessing.image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break  # otherwise the generator would loop indefinitely
# 增加全连接层实现分类
from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(133, activation='softmax'))

model.summary()

from keras import optimizers

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 训练 并保存模型权重
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='dogsclassification.augmentation.model.weights.best.hdf5', verbose=1,
                               save_best_only=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[checkpointer],
    verbose=2)

# 导出模型  测试正确率
model.load_weights('dogsclassification.augmentation.model.weights.best.hdf5')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)

# 画图显示
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

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

model.save('model.h5')
from keras.models import load_model
model = load_model('model.h5')

checkpointer = ModelCheckpoint(filepath='dogsclassification.augmentation.model.weights.best.hdf5', verbose=1,
                               save_best_only=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[checkpointer],
    verbose=2)
# model.save('model.h5')
# model = load_model('model.h5')
# checkpointer = ModelCheckpoint(filepath='dogsclassification.augmentation.model.weights.best.hdf5', verbose=1,
#                                save_best_only=True)
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=100,
#     validation_data=validation_generator,
#     validation_steps=50,
#     callbacks=[checkpointer],
#     verbose=2)
model.save('model.h5')
model.load_weights('dogsclassification.augmentation.model.weights.best.hdf5')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
# 画图显示
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

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

