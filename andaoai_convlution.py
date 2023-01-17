import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import glob
import os
print('Tensorflow version: {}'.format(tf.__version__))
def load_preprosess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [360, 360])
    image = tf.image.random_crop(image, [256, 256, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.cast(image, tf.float32)
    image = image/255
    label = tf.reshape(label, [1])
    return image, label
def load_preprosess_test_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = image/255.0
    label = tf.reshape(label, [1])
    return image, label
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_image_path = glob.glob('../input/cat-dog/dc_2000/train/*/*.jpg')
np.random.shuffle(train_image_path)
train_image_label = [int(path.split('/')[5] == 'cats') for path in train_image_path]
train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
train_image_ds = train_image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
for img, label in train_image_ds.take(1):
    plt.imshow(img)
BATCH_SIZE = 64
train_count = len(train_image_path)
train_image_ds = train_image_ds.shuffle(500).batch(BATCH_SIZE)
train_image_ds = train_image_ds.prefetch(AUTOTUNE)
test_image_path = glob.glob('../input/cat-dog/dc_2000/test/*/*.jpg')
test_image_label = [int(path.split('/')[5] == 'cats') for path in test_image_path]
test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
test_image_ds = test_image_ds.map(load_preprosess_test_image, num_parallel_calls=AUTOTUNE)
test_image_ds = test_image_ds.batch(BATCH_SIZE)
test_image_ds = test_image_ds.prefetch(AUTOTUNE)
model = keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1)
])
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
epoch_loss_avg = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.Accuracy()

epoch_loss_avg_test = tf.keras.metrics.Mean('test_loss')
test_accuracy = tf.keras.metrics.Accuracy()
def train_step(model, images, labels):
    with tf.GradientTape() as t:
        pred = model(images)
        loss_step = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)
    grads = t.gradient(loss_step, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    epoch_loss_avg(loss_step)
    train_accuracy(labels, tf.cast(pred>0, tf.int32))
def test_step(model, images, labels):
    pred = model(images, training=False)
    loss_step = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)
    epoch_loss_avg_test(loss_step)
    test_accuracy(labels, tf.cast(pred>0, tf.int32))
train_loss_results = []
train_acc_results = []

test_loss_results = []
test_acc_results = []
num_epochs = 2
for epoch in range(num_epochs):
    for imgs_, labels_ in train_image_ds:
        train_step(model, imgs_, labels_)
        print('.', end='')
    print()
    
    train_loss_results.append(epoch_loss_avg.result())
    train_acc_results.append(train_accuracy.result())
    
    
    for imgs_, labels_ in test_image_ds:
        test_step(model, imgs_, labels_)
        
    test_loss_results.append(epoch_loss_avg_test.result())
    test_acc_results.append(test_accuracy.result())
    
    print('Epoch:{}: loss: {:.3f}, accuracy: {:.3f}, test_loss: {:.3f}, test_accuracy: {:.3f}'.format(
        epoch + 1,
        epoch_loss_avg.result(),
        train_accuracy.result(),
        epoch_loss_avg_test.result(),
        test_accuracy.result()
    ))
    
    epoch_loss_avg.reset_states()
    train_accuracy.reset_states()
    
    epoch_loss_avg_test.reset_states()
    test_accuracy.reset_states()
plt.plot(range(1,num_epochs+1),train_loss_results,label='train_loss')
plt.plot(range(1,num_epochs+1),test_loss_results,label='test_loss')
plt.grid(True)
plt.legend()
plt.plot(range(1,num_epochs+1),train_acc_results,label='train_accuracy')
plt.plot(range(1,num_epochs+1),test_acc_results,label='test_accuracy')
plt.grid(True)
plt.legend()
import random
i = random.randint(0,64)
for img, musk in train_image_ds.take(1):
    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[i]))
    a=model.predict(img)
    print(a[i])
    print(a.shape)
convlution_input = model.input
convlution_output_0 = model.layers[3].output
convlution_output_1 = model.layers[8].output
convlution_output_2 = model.layers[13].output
middle_convlution_model = tf.keras.models.Model(inputs = convlution_input,
                                                outputs = [convlution_output_0,convlution_output_1,convlution_output_2])
import random
random_image = random.randint(0,64)
output_convlution_one = 0
output_convlution_two = 0
output_convlution_three = 0
for img, musk in train_image_ds.take(1):
    plt.subplot(2,1,1)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[random_image]))
    a=middle_convlution_model.predict(img)
    print(a[0].shape)
    print(a[1].shape)
    print(a[2].shape)
    #plt.subplot(2,1,2)
    output_convlution_one = a[0]
    output_convlution_two = a[1]
    output_convlution_three = a[2]
k=8
plt.figure(figsize=(30, 30))
for i in range(1,k*k):
    plt.subplot(k,k,i)
    plt.imshow(output_convlution_one[random_image,:,:,i])
    plt.axis('off')
k=11
plt.figure(figsize=(30, 30))
for i in range(1,k*k):
    plt.subplot(k,k,i)
    plt.imshow(output_convlution_two[random_image,:,:,i])
    plt.axis('off')
k=16
plt.figure(figsize=(20, 20))
for i in range(1,k*k):
    plt.subplot(k,k,i)
    plt.imshow(output_convlution_three[random_image,:,:,i])
    plt.axis('off')