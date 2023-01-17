import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import tensorflow.data as td

import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler
AUTOTUNE = tf.data.experimental.AUTOTUNE
df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')

df.sample(5)
df.id_code.describe()
df.isna().sum()
df.plot.hist(by='diagnosis')

plt.show()
oversampler = RandomOverSampler()
x,y = oversampler.fit_resample(df.id_code.values.reshape(-1,1),df.diagnosis.values)
df=pd.DataFrame({"id_code":x.flatten(),"diagnosis":y})
df.plot.hist(by='diagnosis')

plt.show()
imagePaths = df.apply(lambda x: '/kaggle/input/aptos2019-blindness-detection/train_images/'+str(x[0])+'.png',axis=1).values

classes = df.iloc[:,1].values
imagePaths[:10]
classes[:10]
def load(path):

    image = tf.image.decode_png(tf.io.read_file(path),channels=3)

    return image

with tf.Session() as sess:

    image,label = sess.run(load(imagePaths[0])),classes[0]

plt.imshow(image)

plt.title('Diagnosis: '+str(label))

plt.show()
pathDS = td.Dataset.from_tensor_slices(imagePaths)
labelDS = td.Dataset.from_tensor_slices(classes)
def oneHotter(label):

    return tf.one_hot(label,5)



oneHotLabelDS = labelDS.map(oneHotter,num_parallel_calls=AUTOTUNE)
imageDSIterator = pathDS.map(load,num_parallel_calls=AUTOTUNE).make_one_shot_iterator()

elem = imageDSIterator.get_next()

with tf.Session() as sess:

    plt.figure(figsize=(40,30))

    for idx in range(12):

        image=sess.run(elem)

        plt.subplot(3,4,idx+1)

        plt.imshow(image)

        plt.grid(False)

        plt.xticks([])

        plt.yticks([])

    plt.show()
def transform_perspective(image):

    def x_y_1():

        x = tf.random_uniform([], minval=-0.3, maxval=-0.15)

        y = tf.random_uniform([], minval=-0.3, maxval=-0.15)

        return x, y

     

    def x_y_2():

        x = tf.random_uniform([], minval=0.15, maxval=0.3)

        y = tf.random_uniform([], minval=0.15, maxval=0.3)

        return x, y       



    def trans(image):

        ran = tf.random_uniform([])

        x = tf.random_uniform([], minval=-0.3, maxval=0.3)

        x_com = tf.random_uniform([], minval=1-x-0.1, maxval=1-x+0.1)



        y = tf.random_uniform([], minval=-0.3, maxval=0.3)

        y_com = tf.random_uniform([], minval=1-y-0.1, maxval=1-y+0.1)



        transforms =  [x_com, x,0,y,y_com,0,0.00,0]



        ran = tf.random_uniform([]) 

        image = tf.cond(ran<0.5, lambda:tf.contrib.image.transform(image,transforms,interpolation='NEAREST', name=None), 

                lambda:tf.contrib.image.transform(image,transforms,interpolation='BILINEAR', name=None))

        return image



    ran = tf.random_uniform([])

    image = tf.cond(ran<1, lambda: trans(image), lambda:image)



    return image
def loadAndPreProcess(path):

    image = tf.image.decode_png(tf.io.read_file(path),channels=3)

    image = tf.image.resize(image,[299,299])

    image = tf.image.random_brightness(image,0.5)

    image = tf.image.random_hue(image,0.05)

    image = tf.image.random_contrast(image,0.75,1.25)

    image = tf.image.random_saturation(image,0.75,1.25)

    image = tf.image.random_flip_left_right(image)

    image = tf.contrib.image.rotate(image,tf.random_uniform(shape=[], minval=-15, maxval=15, dtype=tf.float32))

    image = transform_perspective(image)

    image /= 255.

    image-=0.5

    image*=2.

    return image

    

augImageDS = pathDS.map(loadAndPreProcess,num_parallel_calls=AUTOTUNE)

augImageDSIterator = augImageDS.make_one_shot_iterator()

elem = augImageDSIterator.get_next()

with tf.Session() as sess:

    plt.figure(figsize=(40,30))

    for idx in range(12):

        image = sess.run(elem)

        plt.subplot(3,4,idx+1)

        plt.imshow(image/2+0.5)

        plt.grid(False)

        plt.xticks([])

        plt.yticks([])

    plt.show()
dataset = td.Dataset.zip((augImageDS,oneHotLabelDS))
batchSize=2

ds = dataset.shuffle(buffer_size=len(imagePaths)//10)

ds = ds.repeat()

ds = ds.batch(batchSize)

ds = ds.prefetch(buffer_size=AUTOTUNE)

ds = ds.make_one_shot_iterator()
print(ds)
base = tf.keras.applications.InceptionResNetV2(input_shape=(299,299,3),include_top=False,weights='imagenet')
base.trainable=False

base.summary()
model = tf.keras.Sequential([base,

                             tf.keras.layers.GlobalAveragePooling2D(),

                             tf.keras.layers.Dense(100,activation='relu'),

                             tf.keras.layers.Dense(100,activation='relu'),

                             tf.keras.layers.Dense(5)])



model.compile(optimizer=tf.train.AdamOptimizer(),loss=tf.losses.softmax_cross_entropy,metrics=['accuracy'])




model.fit(ds,epochs=20,verbose=1,steps_per_epoch=len(imagePaths)//batchSize)
model.save('EyeClassifier')