import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import os

import glob



tf.__version__
os.listdir(r'../input/oxfordiitpetdatasetfromciel/Oxford-IIT-Pet/annotations/trimaps')[-5:]
my_img = tf.io.read_file(r'../input/oxfordiitpetdatasetfromciel/Oxford-IIT-Pet/annotations/trimaps/Ragdoll_85.png')

my_img = tf.image.decode_png(my_img)

my_img.shape
my_img = tf.squeeze(my_img)

my_img.shape
plt.imshow(my_img)
my_image = tf.io.read_file(r'../input/oxfordiitpetdatasetfromciel/Oxford-IIT-Pet/images/Ragdoll_85.jpg')

my_image = tf.image.decode_jpeg(my_image)

plt.imshow(my_image)
my_img.numpy().max(), my_img.numpy().min()
np.unique(my_img.numpy())
images = glob.glob(r'../input/oxfordiitpetdatasetfromciel/Oxford-IIT-Pet/images/*.jpg')

len(images)
anno = glob.glob(r'../input/oxfordiitpetdatasetfromciel/Oxford-IIT-Pet/annotations/trimaps/*.png')

images[:3] + anno[:3]
anno_names = [path.split('/trimaps/')[1] for path in anno]

img_names = [path.split('/images/')[1] for path in images]

img_names[:3] + anno_names[:3]
anno_names.sort()

img_names.sort()
img_names[:3] + anno_names[:3]
anno = ['../input/oxfordiitpetdatasetfromciel/Oxford-IIT-Pet/annotations/trimaps/' + name for name in anno_names]

images = ['../input/oxfordiitpetdatasetfromciel/Oxford-IIT-Pet/images/' + name for name in img_names]

images[:3] + anno[:3]
np.random.seed(2020)

index = np.random.permutation(7390)

index
images = np.array(images)[index]

anno = np.array(anno)[index]



ds = tf.data.Dataset.from_tensor_slices((images, anno))



train_ds = ds.skip(1500)

test_ds = ds.take(1500)
def read_jpg(path):

    img = tf.io.read_file(path)

    img = tf.image.decode_jpeg(img, channels=3)

    return img

    

def read_png(path):

    img = tf.io.read_file(path)

    img = tf.image.decode_png(img, channels=1)

    return img
def normalize_img(input_img, input_anno):

    input_img = tf.cast(input_img, tf.float32)

    input_img = input_img/127.5-1

    input_anno -= 1

    return input_img, input_anno
def load_img(input_img, input_anno):

    input_img = read_jpg(input_img)

    input_anno = read_png(input_anno)

    input_img = tf.image.resize(input_img, (224, 224))

    input_anno = tf.image.resize(input_anno, (224, 224))

    return normalize_img(input_img, input_anno)
train_ds = train_ds.map(load_img,

                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_ds = test_ds.map(load_img,

                      num_parallel_calls=tf.data.experimental.AUTOTUNE)



train_ds = train_ds.repeat().shuffle(100).batch(8)

test_ds = test_ds.batch(8)



train_ds
for img, anno in train_ds.take(1):

    plt.subplot(1, 2, 1)

    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))

    plt.subplot(1, 2, 2)

    plt.imshow(tf.keras.preprocessing.image.array_to_img(anno[0]))
conv_base = tf.keras.applications.VGG16(weights='imagenet',

                                        input_shape=(224, 224, 3),

                                        include_top=False)

conv_base.summary()
layer_names = ['block5_conv3',

               'block4_conv3',

               'block3_conv3',

               'block5_pool']



outputs = [conv_base.get_layer(name).output for name in layer_names]



multi_output_model = tf.keras.models.Model(inputs=conv_base.input,

                                           outputs=outputs)
multi_output_model.trainable = False



inputs = tf.keras.layers.Input(shape=(224, 224, 3))



out_block5_conv3, out_block4_conv3, out_block3_conv3, out = multi_output_model(inputs)



out.shape
out_block5_conv3.shape
x1 = tf.keras.layers.Conv2DTranspose(512, 3,

                                     strides=2, 

                                     padding='same',

                                     activation='relu')(out)



x1.shape
x1 = tf.keras.layers.Conv2D(512, 3,

                            padding='same',

                            activation='relu')(x1)



x1.shape
x2 = tf.add(x1, out_block5_conv3)

x2.shape
x2 = tf.keras.layers.Conv2DTranspose(512, 3,

                                     strides=2, 

                                     padding='same',

                                     activation='relu')(x2)

x2 = tf.keras.layers.Conv2D(512, 3,

                            padding='same',

                            activation='relu')(x2)



x3 = tf.add(x2, out_block4_conv3)



x3 = tf.keras.layers.Conv2DTranspose(256, 3,

                                     strides=2, 

                                     padding='same',

                                     activation='relu')(x3)

x3 = tf.keras.layers.Conv2D(256, 3,

                            padding='same',

                            activation='relu')(x3)



x4 = tf.add(x3, out_block3_conv3)



x4.shape
x5 = tf.keras.layers.Conv2DTranspose(64, 3,

                                     strides=2, 

                                     padding='same',

                                     activation='relu')(x4)



x5 = tf.keras.layers.Conv2D(64, 3,

                            padding='same',

                            activation='relu')(x5)



prediction = tf.keras.layers.Conv2DTranspose(3, 3,

                                     strides=2, 

                                     padding='same',

                                     activation='softmax')(x5)
model = tf.keras.models.Model(inputs=inputs,

                              outputs=prediction)

model.summary()
from keras.utils.vis_utils import plot_model



plot_model(model, to_file='semantic_seg_fcn.png', show_shapes=True, show_layer_names=True)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])



history = model.fit(train_ds, epochs=5, steps_per_epoch=(7390-1500)//8, validation_data=test_ds, validation_steps=1500//8)
loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(5)



plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'bo', label='Validation Loss')

plt.title('Training & Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss Value')

plt.ylim([0, 1])

plt.legend()

plt.show()
n = 3



for img, mask in test_ds.take(1):

    

    pred_mask = model(img)

    pred_mask = tf.argmax(pred_mask, axis=-1)

    # ... -> take all the dimensions of the original pred_mask, tf.newaxis -> expand to (224,224, 1)

    pred_mask = pred_mask[..., tf.newaxis]

    

    plt.figure(figsize=(10, 10))

    

    for i in range(n):

        plt.subplot(n, 3, i*n+1)

        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[i]))

        plt.subplot(n, 3, i*n+2)

        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask[i]))

        plt.subplot(n, 3, i*n+3)

        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[i]))