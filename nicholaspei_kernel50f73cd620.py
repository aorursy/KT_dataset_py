!pip install  efficientnet
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import glob
import os
import pandas as pd
import efficientnet.tfkeras as efn
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
gpu
tf.config.experimental.set_memory_growth(gpu[0], True)
AUTOTUNE = tf.data.experimental.AUTOTUNE
print(os.listdir('/kaggle/input/foot-challenge2'))
train_image_path = glob.glob('/kaggle/input/foot-challenge2/train/*.jpg')
label_data = pd.read_csv('/kaggle/input/foot-challenge2/train.csv')
labels = [label_data['label'][int(index.split('/')[-1].split('.')[0])] for index in train_image_path]
image_ds = tf.data.Dataset.from_tensor_slices((train_image_path,labels))
def load_preprocess_image(path,label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,[300,300])
    image = tf.image.random_crop(image,[260,260,3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image,0.2)
    
    image = tf.cast(image,tf.float32)
    image = image/255
    label = tf.reshape(label,[1])
    return image,label
def load_preprocess_image_test(path,label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,[260,260])
    image = tf.cast(image,tf.float32)
    image = image/255
    label = tf.reshape(label,[1])
    return image,label
val_count = int(len(labels)*0.2)
train_count = len(labels)-val_count
val_count,train_count
image_train_ds = image_ds.skip(val_count)
image_val_ds = image_ds.take(val_count)
image_train_ds = image_train_ds.map(load_preprocess_image,num_parallel_calls=AUTOTUNE)
image_val_ds = image_val_ds.map(load_preprocess_image_test,num_parallel_calls=AUTOTUNE)
image_train_ds
BATCH_SIZE = 32
image_train_ds = image_train_ds.repeat().shuffle(train_count).batch(BATCH_SIZE)
image_train_ds = image_train_ds.prefetch(AUTOTUNE)#预取
image_val_ds = image_val_ds.batch(BATCH_SIZE)
image_val_ds = image_val_ds.prefetch(AUTOTUNE)#预取
# covn_base = keras.applications.VGG19(weights='imagenet',
#                                            input_shape=(256,256,3),
#                                            include_top=False,
#                                            pooling='avg')
covn_base = efn.EfficientNetB2(weights='imagenet',
                               input_shape=(260,260,3),
                               include_top=False,
                               pooling='avg')
covn_base.summary()
model = keras.Sequential()
model.add(covn_base)
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1024,activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(512,activation='relu'))
model.add(keras.layers.Dense(4,activation='sigmoid'))
model.summary()
covn_base.trainable = False #设置权重参数不可训练
model.summary()
#编译
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
             loss = 'sparse_categorical_crossentropy',
             metrics=['acc'])
history = model.fit(
    image_train_ds,
    steps_per_epoch=train_count//BATCH_SIZE,
    epochs=8,
    validation_data=image_val_ds,
    validation_steps=val_count//BATCH_SIZE
)
plt.plot(history.epoch,history.history.get('loss'),label='loss')
plt.plot(history.epoch,history.history.get('val_loss'),label='val_loss')
plt.legend()
plt.plot(history.epoch,history.history.get('acc'),label='acc')
plt.plot(history.epoch,history.history.get('val_acc'),label='val_acc')
plt.legend()
model.save('mstz_model_EfficientNetB2.h5')
test_image_path = glob.glob('/kaggle/input/foot-challenge2/test/*.jpg')
test_image_path.sort(key=lambda x:int(x.split('/')[-1].split('.')[0]))
#加载图片
def load_preprocess_images(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,[260,260])
    image = tf.cast(image,tf.float32)
    image = image/255
    image = tf.expand_dims(image,0)
    return image

images = [load_preprocess_images(image_path) for image_path in test_image_path]

len(images)
image_count = len(images)
values = []
result_dict = {}

for i in range(image_count):
    pred = model.predict(images[i])
    values.append(np.argmax(pred))
    print('.',end='')
#构建字典
for i in range(image_count):
    result_dict[i]=values[i]
#写文件
with open('result_efficientB2.csv','w',encoding='utf-8') as f:
    [f.write('{0},{1}\n'.format(key, value)) for (key,value) in result_dict.items()]