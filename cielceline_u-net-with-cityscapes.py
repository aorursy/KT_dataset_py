import tensorflow as tf

import numpy as np

import os

import glob

import matplotlib.pyplot as plt

%matplotlib inline
img = glob.glob('../input/cityscapes/Cityspaces/images/train/*/*.png') # tf.io.glob.glob

label = glob.glob('../input/cityscapes/Cityspaces/gtFine/train/*/*_gtFine_labelIds.png')

img_names = [path.split('/train/')[1].split('_leftImg8bit.png')[0] for path in img]

label = ['../input/cityscapes/Cityspaces/gtFine/train/' + name + '_gtFine_labelIds.png' for name in img_names]



len(img)
index = np.random.permutation(2975)

img = np.array(img)[index]

label = np.array(label)[index]



img[:5], label[:5]
val_img = glob.glob('../input/cityscapes/Cityspaces/images/val/*/*.png') # tf.io.glob.glob

val_label = glob.glob('../input/cityscapes/Cityspaces/gtFine/val/*/*_gtFine_labelIds.png')

img_names = [path.split('/val/')[1].split('_leftImg8bit.png')[0] for path in val_img]

val_label = ['../input/cityscapes/Cityspaces/gtFine/val/' + name + '_gtFine_labelIds.png' for name in img_names]



len(val_img), len(val_label)
train_ds = tf.data.Dataset.from_tensor_slices((img, label))

val_ds = tf.data.Dataset.from_tensor_slices((val_img, val_label))
def read_png(img):

    img = tf.io.read_file(img)

    img = tf.image.decode_png(img, channels=3)

    return img

    

def read_png_label(img):

    img = tf.io.read_file(img)

    img = tf.image.decode_png(img, channels=1)

    return img
def rand_crop(img, label):

    concat_img = tf.concat([img, label], axis=-1)

    concat_img = tf.image.resize(concat_img, [280, 560], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    crop_img = tf.image.random_crop(concat_img, [256, 256, 4])

    return crop_img[:, :, :3], crop_img[:, :, 3:]



def norm(img, label):

    img = tf.cast(img, tf.float32)/127.5-1

    label = tf.cast(label, tf.int32)

    return img, label
def load_img_train(img, label):

    img = read_png(img)

    label = read_png_label(label)

    

    img, label = rand_crop(img, label)

    

    if tf.random.uniform(()) > 0.5:

        img = tf.image.flip_left_right(img)

        label = tf.image.flip_left_right(label)

    return norm(img, label)



def load_img_val(img, label):

    img = read_png(img)

    label = read_png_label(label)

    

    img = tf.image.resize(img, [256, 256])

    label = tf.image.resize(label, [256, 256])

    return norm(img, label)
BATCH_SIZE = 32

BUFFER_SIZE = 300

steps_per_epoch = 2975 // BATCH_SIZE

validation_steps = 500 // BATCH_SIZE

auto = tf.data.experimental.AUTOTUNE



train_ds = train_ds.map(load_img_train, num_parallel_calls=auto)

val_ds = val_ds.map(load_img_val, num_parallel_calls=auto)
for img, label in val_ds.take(1):

    plt.subplot(1, 2, 1)

    plt.imshow((img + 1)/2)

    plt.subplot(1, 2, 2)

    plt.imshow(np.squeeze(label))
train_ds = train_ds.cache().repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(auto)

val_ds = val_ds.cache().batch(BATCH_SIZE)
def create_model():

    inputs = tf.keras.layers.Input(shape=(256, 256, 3))

    

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    x = tf.keras.layers.BatchNormalization()(x) # 256*256*64

    

    x1 = tf.keras.layers.MaxPooling2D(padding='same')(x) # 128*128*64

    

    x1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x1)

    x1 = tf.keras.layers.BatchNormalization()(x1)

    x1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x1)

    x1 = tf.keras.layers.BatchNormalization()(x1)  # 128*128*128

    

    x2 = tf.keras.layers.MaxPooling2D(padding='same')(x1) # 64*64*128

    

    x2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x2)

    x2 = tf.keras.layers.BatchNormalization()(x2)

    x2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x2)

    x2 = tf.keras.layers.BatchNormalization()(x2)  # 64*64*256

    

    x3 = tf.keras.layers.MaxPooling2D(padding='same')(x2) # 32*32*256

    

    x3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x3)

    x3 = tf.keras.layers.BatchNormalization()(x3)

    x3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x3)

    x3 = tf.keras.layers.BatchNormalization()(x3)  # 32*32*512

    

    x4 = tf.keras.layers.MaxPooling2D(padding='same')(x3) # 16*16*512

    

    x4 = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')(x4)

    x4 = tf.keras.layers.BatchNormalization()(x4)

    x4 = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')(x4)

    x4 = tf.keras.layers.BatchNormalization()(x4)  # 16*16*1024

    

    x5 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2, padding='same', activation='relu')(x4)

    x5 = tf.keras.layers.BatchNormalization()(x5)  # 32*32*512

    

    x6 = tf.concat([x3, x5], axis=-1) # 32*32*1024

    

    x6 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x6)

    x6 = tf.keras.layers.BatchNormalization()(x6)

    x6 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x6)

    x6 = tf.keras.layers.BatchNormalization()(x6)  # 32*32*512

    

    x7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same', activation='relu')(x6)

    x7 = tf.keras.layers.BatchNormalization()(x7)  # 64*64*256

    

    x8 = tf.concat([x2, x7], axis=-1) # 64*64*512

    

    x8 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x8)

    x8 = tf.keras.layers.BatchNormalization()(x8)

    x8 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x8)

    x8 = tf.keras.layers.BatchNormalization()(x8)  # 64*64*256

    

    x9 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same', activation='relu')(x8)

    x9 = tf.keras.layers.BatchNormalization()(x9)  # 128*128*128

    

    x10 = tf.concat([x1, x9], axis=-1) # 128*128*256

    

    x10 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x10)

    x10 = tf.keras.layers.BatchNormalization()(x10)

    x10 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x10)

    x10 = tf.keras.layers.BatchNormalization()(x10)  # 128*128*128

    

    x11 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same', activation='relu')(x10)

    x11 = tf.keras.layers.BatchNormalization()(x11)  # 256*256*64

    

    x12 = tf.concat([x, x11], axis=-1) # 256*256*128

    

    x12 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x12)

    x12 = tf.keras.layers.BatchNormalization()(x12)

    x12 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x12)

    x12 = tf.keras.layers.BatchNormalization()(x12)  # 256*256*64

    

    outputs = tf.keras.layers.Conv2D(34, 1, activation='softmax')(x12) # 256*256*34

    

    return tf.keras.Model(inputs=inputs, outputs=outputs)
model = create_model()

model.summary()
tf.keras.utils.plot_model(model)
# tf.keras.metrics.MeanIoU(num_classes=34) <- One-Hot Coding



class MeanIoU(tf.keras.metrics.MeanIoU):

    def __call__(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.argmax(y_pred, axis=-1)

        return super().__call__(y_true, y_pred, sample_weight=sample_weight)
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy', 

              metrics=['acc'])
history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=validation_steps, epochs=50)
loss = history.history['loss']

val_loss = history.history['val_loss']



plt.figure()

plt.plot(range(50), loss, 'r', label='Training Loss')

plt.plot(range(50), val_loss, 'bo', label='Validation Loss')

plt.title('Training & Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')



plt.legend()

plt.show()
num = 3



for img, label in val_ds.take(1):

    pred_label = model.predict(img)

    pred_label = tf.argmax(pred_label, axis=-1)

    pred_label = pred_label[..., tf.newaxis]

    

    plt.figure(figsize=(10, 10))

    for i in range(num):

        plt.subplot(num, 3, i*num+1)

        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[i]))

        plt.subplot(num, 3, i*num+2)

        plt.imshow(tf.keras.preprocessing.image.array_to_img(label[i]))

        plt.subplot(num, 3, i*num+3)

        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_label[i]))
for img, label in train_ds.take(1):

    pred_label = model.predict(img)

    pred_label = tf.argmax(pred_label, axis=-1)

    pred_label = pred_label[..., tf.newaxis]

    

    plt.figure(figsize=(10, 10))

    for i in range(num):

        plt.subplot(num, 3, i*num+1)

        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[i]))

        plt.subplot(num, 3, i*num+2)

        plt.imshow(tf.keras.preprocessing.image.array_to_img(label[i]))

        plt.subplot(num, 3, i*num+3)

        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_label[i]))
model.save('MyUNet.h5')