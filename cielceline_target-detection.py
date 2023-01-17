import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline



# Analyse xml files in annotations e.g. web crawling

from lxml import etree

import numpy as np

import glob

from matplotlib.patches import Rectangle
image_path = glob.glob('../input/oxfordiitpetdatasetfromciel/Oxford-IIT-Pet/images/*.jpg')

image_path[:5] + image_path[-5:]
len(image_path)
xml_path = glob.glob('../input/oxfordiitpetdatasetfromciel/Oxford-IIT-Pet/annotations/xmls/*.xml')
len(xml_path)
xml_path[:3] + xml_path[-3:]
xml_names = [path.split('xmls/')[1].split('.xml')[0] for path in xml_path]

xml_names[:3]
img_train = ['../input/oxfordiitpetdatasetfromciel/Oxford-IIT-Pet/images/' + name + '.jpg' for name in xml_names]

img_train[:3]
img_test = [path for path in image_path if path not in img_train]

len(img_test)
def to_labels(path):

    xml = open(path).read()

    sel = etree.HTML(xml)

    width = int(sel.xpath('//size/width/text()')[0])

    height = int(sel.xpath('//size/height/text()')[0])

    x_min = int(sel.xpath('//bndbox/xmin/text()')[0])

    x_max = int(sel.xpath('//bndbox/xmax/text()')[0])

    y_min = int(sel.xpath('//bndbox/ymin/text()')[0])

    y_max = int(sel.xpath('//bndbox/ymax/text()')[0])

    return [x_min/width, y_min/height, x_max/width, y_max/height]
labels = [to_labels(path) for path in xml_path]
labels[:3]
out1, out2, out3, out4 = zip(*labels)

out1[:5]
label_ds = tf.data.Dataset.from_tensor_slices((np.array(out1), np.array(out2), np.array(out3), np.array(out4)))

label_ds
def load_img(path):

    img = tf.io.read_file(path)

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, [224, 224])

    img = img/127.5 - 1

    return img
img_ds = tf.data.Dataset.from_tensor_slices(img_train)

img_ds = img_ds.map(load_img)

img_ds
ds = tf.data.Dataset.zip((img_ds, label_ds))

ds
ds = ds.repeat().shuffle(3686).batch(32)

ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
for img, label in ds.take(3):

    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))

    out1, out2, out3, out4 = label

    x_min, y_min, x_max, y_max = out1[0]*224, out2[0]*224, out3[0]*224, out4[0]*224

    rect = Rectangle((x_min, y_min), (x_max - x_min), (y_max - y_min), fill=False, color='blue')

    ax = plt.gca()

    ax.axes.add_patch(rect)

    plt.show()
xception = tf.keras.applications.Xception(weights='imagenet',

                                          include_top=False,

                                          input_shape=(224, 224, 3))
inputs = tf.keras.layers.Input(shape=(224, 224, 3))

x = xception(inputs)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(2048, activation='relu')(x)

x = tf.keras.layers.Dense(256, activation='relu')(x)



out1 = tf.keras.layers.Dense(1)(x)

out2 = tf.keras.layers.Dense(1)(x)

out3 = tf.keras.layers.Dense(1)(x)

out4 = tf.keras.layers.Dense(1)(x)



outs = [out1, out2, out3, out4]



model = tf.keras.models.Model(inputs=inputs, outputs=outs)
model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001),

              loss='mse', 

              metrics=['mae'])
test_ds = ds.take(640)

train_ds = ds.skip(640)

steps_per_epoch = (3686-640)//32

validation_steps = 640//32
history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, 

                    validation_data=test_ds, validation_steps=validation_steps,

                    epochs=10)
loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(10)



plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'bo', label='Validation Loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.show()
model.save('detect_target.h5')

new_model = tf.keras.models.load_model('detect_target.h5')
plt.figure(figsize=(8, 24))



test_images = tf.data.Dataset.from_tensor_slices(img_test)

test_images = test_images.map(load_img)

test_images = test_images.batch(32)



for imgs in test_images.take(1):

    out1, out2, out3, out4 = new_model.predict(imgs)

    for i in range(3):

        plt.subplot(3, 1, i+1)

        plt.imshow(tf.keras.preprocessing.image.array_to_img(imgs[i]))

        x_min, y_min, x_max, y_max = out1[i]*224, out2[i]*224, out3[i]*224, out4[i]*224

        ax = plt.gca()

        rect = Rectangle((x_min, y_min), (x_max - x_min), (y_max - y_min), fill=False, color='blue')

        ax.axes.add_patch(rect)