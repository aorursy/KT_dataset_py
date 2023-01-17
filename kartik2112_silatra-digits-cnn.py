import tensorflow as tf

import keras

import os

import shutil

import cv2

from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import numpy as np
img_path = '../input/indian-sign-language-translation-letters-n-digits/'



for class1 in os.listdir(img_path):

    num_images = len(os.listdir(os.path.join(img_path,class1)))

    for (n,filename) in enumerate(os.listdir(os.path.join(img_path,class1))):

        img = os.path.join(img_path,class1,filename)

        if not os.path.exists('test/'+class1+'/'):

            os.makedirs('test/'+class1+'/')

            os.makedirs('train/'+class1+'/')

            os.makedirs('val/'+class1+'/')

        if n < int(0.1 * num_images):

            shutil.copy(img,'test/'+class1+'/'+filename)

        elif n < int(0.8 * num_images):

            shutil.copy(img,'train/'+class1+'/'+filename)

        else:

            shutil.copy(img,'val/'+class1+'/'+filename)
def load_data(image_dir):

    images = []

    y = []

    classNum = 0

    for class1 in tqdm(os.listdir(image_dir)):

        for file_name in os.listdir(os.path.join(image_dir,class1)):

            images.append(cv2.imread(os.path.join(image_dir,class1,file_name)))

            y.append(classNum)

        classNum += 1

    print(f'Loaded {len(images)} images from {image_dir} directory')

    images = np.array(images)

    y = np.array(y)

    return images,y
_, _ = load_data('train/')

test_images,test_labels = load_data('test/')

_, _ = load_data('val/')
arr = [i for i in range(10)] + [chr(ord('a')+i) for i in range(26)]

arr.remove('v')

arr.remove('h')

arr.remove('j')

label_dicts = {i:arr[i] for i in range(len(arr))}

label_dicts
def show_samples(X,y,n=30):

    classes = np.unique(y)

    classNo = 0

    for class1 in classes:

        imgs = X[y == class1][:n]

        j = 10

        i = n // 10

        plt.figure(figsize=(15,1))

        for (c,img) in enumerate(imgs,1):

            plt.subplot(i,j,c)

            plt.imshow(img)

            plt.xticks([])

            plt.yticks([])

        plt.suptitle(f'Digit/Letter: {label_dicts[class1]}')

        classNo += 1

        if classNo == 15: 

            break
datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.05,

        zoom_range=(0.8,1.2),

        width_shift_range=0.05,

        height_shift_range=0.05,

        rotation_range=30,

        brightness_range=(0.5,1.5),

        channel_shift_range=1,

        horizontal_flip=True)



train_generator = datagen.flow_from_directory(

        'train/',

        target_size=(224,224),

        batch_size=350,

        class_mode='categorical')

valid_generator = datagen.flow_from_directory(

        'val/',

        target_size=(224,224),

        batch_size=100,

        class_mode='categorical')

# plt.imshow(train_generator.next()[0].reshape(140,160,3))
batch1 = train_generator.next()

show_samples(batch1[0], batch1[1].argmax(axis=1),n=10)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
model = Sequential()



model.add(Conv2D(16,kernel_size=8,activation='relu',input_shape=(224,224,3),padding='same'))

model.add(MaxPooling2D(pool_size=(8,8),strides=(8,8),padding='same'))

model.add(Conv2D(32,kernel_size=3,activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(4,4),strides=(4,4),padding='same'))

model.add(Conv2D(64,kernel_size=3,activation='relu',padding='same'))

model.add(Flatten())

model.add(Dense(len(label_dicts),activation='softmax'))
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(

    initial_learning_rate=0.01,

    decay_steps=8000,

    decay_rate=0.01)

opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
train_step_size = train_generator.n//train_generator.batch_size

val_step_size = valid_generator.n//valid_generator.batch_size

history = model.fit_generator(

        train_generator,

        steps_per_epoch=train_step_size,

        validation_data=valid_generator,

        validation_steps=val_step_size,

        epochs=15)
# plot model performance

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs_range = range(1, len(history.epoch) + 1)



plt.figure(figsize=(15,5))



plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Train Set')

plt.plot(epochs_range, val_acc, label='Val Set')

plt.legend(loc="best")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Model Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Train Set')

plt.plot(epochs_range, val_loss, label='Val Set')

plt.legend(loc="best")

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Model Loss')



plt.tight_layout()

plt.show()
model.evaluate_generator(valid_generator)
model.save('Silatra_HandPoses_CNN_27May20.h5')