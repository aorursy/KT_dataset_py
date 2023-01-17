import tensorflow as tf

import numpy as np

import os

%matplotlib inline
os.listdir('../input/cityscapes/Cityspaces/images')
images = tf.io.gfile.glob("../input/cityscapes/Cityspaces/images/train/*/*.png")

len(images)
annotations = tf.io.gfile.glob("../input/cityscapes/Cityspaces/gtFine/train/*/*_gtFine_labelIds.png")

len(annotations)
images.sort(key=lambda x:x.split('train/')[-1].split('_leftImg8bit.png')[0])

annotations.sort(key=lambda x:x.split('train/')[-1].split('_gtFine_labelIds.png')[0])
images[:5]
annotations[:5]
train_count = len(images)

len(images), len(annotations)
np.random.seed(2019)

index = np.random.permutation(len(images))
images = np.array(images)[index]
anno = np.array(annotations)[index]
dataset_train = tf.data.Dataset.from_tensor_slices((images, anno))
images_val = tf.io.gfile.glob("../input/cityscapes/Cityspaces/images/val/*/*.png")

annotations_val = tf.io.gfile.glob("../input/cityscapes/Cityspaces/gtFine/val/*/*_gtFine_labelIds.png")
val_count = len(images_val)

len(images_val), len(annotations_val)
dataset_val = tf.data.Dataset.from_tensor_slices((images_val, annotations_val))
def read_png(path):

    img = tf.io.read_file(path)

    img = tf.image.decode_png(img, channels=3)

    return img
def read_png_label(path):

    img = tf.io.read_file(path)

    img = tf.image.decode_png(img, channels=1)

    return img 
def normalize(input_image, input_mask):

    input_image = tf.cast(input_image, tf.float32)/127.5 - 1

    input_mask = tf.cast(input_mask, tf.int32)

    return input_image, input_mask
IMG_HEIGHT = 256

IMG_WIDTH = 256
def random_crop(img, mask):

    concat_img = tf.concat([img, mask], axis=-1)

    concat_img = tf.image.resize(concat_img, (280, 280),

                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    crop_img = tf.image.random_crop(concat_img, [256, 256, 4])

    return crop_img[ :, :, :3], crop_img[ :, :, 3:]
def load_image_train(input_image_path, input_mask_path):

    input_image = read_png(input_image_path)

    input_mask = read_png_label(input_mask_path)

    

    input_image, input_mask = random_crop(input_image, input_mask)



    if tf.random.uniform(()) > 0.5:

        input_image = tf.image.flip_left_right(input_image)

        input_mask = tf.image.flip_left_right(input_mask)



    input_image, input_mask = normalize(input_image, input_mask)



    return input_image, input_mask
def load_image_val(input_image_path, input_mask_path):

    input_image = read_png(input_image_path)

    input_mask = read_png_label(input_mask_path)

    input_image = tf.image.resize(input_image, (IMG_HEIGHT, IMG_WIDTH))

    input_mask = tf.image.resize(input_mask, (IMG_HEIGHT, IMG_WIDTH))



    input_image, input_mask = normalize(input_image, input_mask)



    return input_image, input_mask
BATCH_SIZE = 32

BUFFER_SIZE = 300

STEPS_PER_EPOCH = train_count // BATCH_SIZE

VALIDATION_STEPS = val_count // BATCH_SIZE 
AUTO = tf.data.experimental.AUTOTUNE
dataset_train = dataset_train.map(load_image_train, num_parallel_calls=AUTO)

dataset_val = dataset_val.map(load_image_val, num_parallel_calls=AUTO)
dataset_train = dataset_train.cache().repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

dataset_val = dataset_val.cache().batch(BATCH_SIZE)  
dataset_train
dataset_val
OUTPUT_CHANNELS = 34
def create_model():

    inputs = tf.keras.layers.Input(shape=(256, 256, 3))

    

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    x = tf.keras.layers.BatchNormalization()(x)     #  256*256*64

    

    x1 = tf.keras.layers.MaxPooling2D(padding='same')(x)   # 128*128*64

    

    x1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x1)

    x1 = tf.keras.layers.BatchNormalization()(x1)     

    x1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x1)

    x1 = tf.keras.layers.BatchNormalization()(x1)     #  128*128*128

    

    x2 = tf.keras.layers.MaxPooling2D(padding='same')(x1)   # 64*64*128

    

    x2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x2)

    x2 = tf.keras.layers.BatchNormalization()(x2)     

    x2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x2)

    x2 = tf.keras.layers.BatchNormalization()(x2)     #  64*64*256

    

    x3 = tf.keras.layers.MaxPooling2D(padding='same')(x2)   # 32*32*256

    

    x3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x3)

    x3 = tf.keras.layers.BatchNormalization()(x3)     

    x3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x3)

    x3 = tf.keras.layers.BatchNormalization()(x3)     #  32*32*512

    

    x4 = tf.keras.layers.MaxPooling2D(padding='same')(x3)   # 16*16*512

    

    x4 = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')(x4)

    x4 = tf.keras.layers.BatchNormalization()(x4)     

    x4 = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')(x4)

    x4 = tf.keras.layers.BatchNormalization()(x4)     #  16*16*1024

    

    #  上采样部分

    

    x5 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2,

                                         padding='same', activation='relu')(x4) 

    x5 = tf.keras.layers.BatchNormalization()(x5)     #  32*32*512

    

    x6 = tf.concat([x3, x5], axis=-1)  #  32*32*1024

    

    x6 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x6)

    x6 = tf.keras.layers.BatchNormalization()(x6)     

    x6 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x6)

    x6 = tf.keras.layers.BatchNormalization()(x6)     #  32*32*512

    

    x7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2,

                                         padding='same', activation='relu')(x6) 

    x7 = tf.keras.layers.BatchNormalization()(x7)     #  64*64*256

    

    x8 = tf.concat([x2, x7], axis=-1)  #  64*64*512

    

    x8 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x8)

    x8 = tf.keras.layers.BatchNormalization()(x8)     

    x8 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x8)

    x8 = tf.keras.layers.BatchNormalization()(x8)     #  64*64*256

    

    x9 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2,

                                         padding='same', activation='relu')(x8) 

    x9 = tf.keras.layers.BatchNormalization()(x9)     #  128*128*128

    

    x10 = tf.concat([x1, x9], axis=-1)  #  128*128*256

    

    x10 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x10)

    x10 = tf.keras.layers.BatchNormalization()(x10)     

    x10 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x10)

    x10 = tf.keras.layers.BatchNormalization()(x10)     #  128*128*128

    

    x11 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2,

                                         padding='same', activation='relu')(x10) 

    x11 = tf.keras.layers.BatchNormalization()(x11)     #  256*256*64

    

    x12 = tf.concat([x, x11], axis=-1)  #  256*256*128

    

    x12 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x12)

    x12 = tf.keras.layers.BatchNormalization()(x12)     

    x12 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x12)

    x12 = tf.keras.layers.BatchNormalization()(x12)     #  256*256*64

    

    output = tf.keras.layers.Conv2D(34, 1, padding='same', activation='softmax')(x12)

    #  256*256*34

    return tf.keras.Model(inputs=inputs, outputs=output)
class MeanIoU(tf.keras.metrics.MeanIoU):

    def __call__(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.argmax(y_pred, axis=-1)

        return super().__call__(y_true, y_pred, sample_weight=sample_weight)
model = create_model()
model.summary()
model.compile(

        optimizer=tf.keras.optimizers.Adam(0.001), 

        loss='sparse_categorical_crossentropy',

        metrics=['acc', MeanIoU(num_classes=34)])
EPOCHS = 60
history = model.fit(dataset_train, 

                    epochs=EPOCHS,

                    steps_per_epoch=STEPS_PER_EPOCH,

                    validation_steps=VALIDATION_STEPS,

                    validation_data=dataset_val)