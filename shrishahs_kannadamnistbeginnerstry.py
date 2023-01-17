# This Python 3 environment comes with many helpful analytics libraries installed# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras as ks

import tensorflow as tf

import glob

from tensorflow import keras

import matplotlib.pyplot as plt

import os

import PIL

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

from tensorflow.keras import layers

import time

from IPython import display



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train=pd.read_csv('../input/Kannada-MNIST/train.csv')

test=pd.read_csv('../input/Kannada-MNIST/test.csv')

submission_sample = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

train_CNN=train

Gtrain=train

label0=Gtrain.groupby('label').get_group(0)

label1=Gtrain.groupby('label').get_group(1)

label2=Gtrain.groupby('label').get_group(2)

label3=Gtrain.groupby('label').get_group(3)

label4=Gtrain.groupby('label').get_group(4)

label5=Gtrain.groupby('label').get_group(5)

label6=Gtrain.groupby('label').get_group(6)

label7=Gtrain.groupby('label').get_group(7)

label8=Gtrain.groupby('label').get_group(8)

label9=Gtrain.groupby('label').get_group(9)

#plt.imshow(test[4][,:,:0])



X_label0=label0.drop('label',axis=1)

X_label1=label1.drop('label',axis=1)

X_label2=label2.drop('label',axis=1)

X_label3=label3.drop('label',axis=1)

X_label4=label4.drop('label',axis=1)

X_label5=label5.drop('label',axis=1)

X_label6=label6.drop('label',axis=1)

X_label7=label7.drop('label',axis=1)

X_label8=label8.drop('label',axis=1)

X_label9=label9.drop('label',axis=1)



Y_label0=label0.label

Y_label1=label1.label

Y_label2=label2.label

Y_label3=label3.label

Y_label4=label4.label

Y_label5=label5.label

Y_label6=label6.label

Y_label7=label7.label

Y_label8=label8.label

Y_label9=label9.label



X_label0=np.asarray(X_label0,dtype=np.float32).reshape(-1,28,28,1)

X_label1=np.asarray(X_label1,dtype=np.float32).reshape(-1,28,28,1)

X_label2=np.asarray(X_label2,dtype=np.float32).reshape(-1,28,28,1)

X_label3=np.asarray(X_label3,dtype=np.float32).reshape(-1,28,28,1)

X_label4=np.asarray(X_label4,dtype=np.float32).reshape(-1,28,28,1)

X_label5=np.asarray(X_label5,dtype=np.float32).reshape(-1,28,28,1)

X_label6=np.asarray(X_label6,dtype=np.float32).reshape(-1,28,28,1)

X_label7=np.asarray(X_label7,dtype=np.float32).reshape(-1,28,28,1)

X_label8=np.asarray(X_label8,dtype=np.float32).reshape(-1,28,28,1)

X_label9=np.asarray(X_label9,dtype=np.float32).reshape(-1,28,28,1)

#label1=Gtrain.get_group('1')



#label1=Gtrain.get_group("1")

#print(label9)



#train=np.array(train,dtype=np.float32)



test=test.drop('id',axis=1)

y=train.label.value_counts()



X_train=train.drop('label',axis=1)

Y_train=train.label



X_train=np.array(X_train,dtype=np.float32)

Y_train=np.array(Y_train)

X_train=X_train/255

test=test/255



X_train=X_train.reshape(-1,28,28,1)

new_image_set=X_train

test=test.values.reshape(-1,28,28,1)



#Y_train=to_categorical(Y_train)



X_train,X_test,y_train,y_test=train_test_split(X_train,Y_train,random_state=42,test_size=0.15)



print(Y_label0.shape)
train_CNN_X=train_CNN.drop('label',axis=1)

train_CNN_Y=train.label



train_CNN_X=np.array(train_CNN_X,dtype=np.float32)

train_CNN_Y=np.array(train_CNN_Y)

train_CNN_X=train_CNN_X/255





BUFFER_SIZE = 6000

BATCH_SIZE = 256
#X_label0=np.asarray(X_label0).reshape(-1,28,28)

#train_dataset = tf.data.Dataset.from_tensor_slices(X_label0).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
def make_generator_model():

    model = tf.keras.Sequential()

    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Reshape((7, 7, 256)))

    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size



    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))

    assert model.output_shape == (None, 7, 7, 128)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    assert model.output_shape == (None, 14, 14, 64)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    assert model.output_shape == (None, 28, 28, 1)



    return model
generator = make_generator_model()



noise = tf.random.normal([1, 100])

generated_image = generator(noise, training=False)



plt.imshow(generated_image[0, :, :, 0], cmap='gray')
def make_discriminator_model():

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',

                                     input_shape=[28, 28, 1]))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Flatten())

    model.add(layers.Dense(1))



    return model
discriminator = make_discriminator_model()

decision = discriminator(generated_image)

print (decision)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss
def generator_loss(fake_output):

    return cross_entropy(tf.ones_like(fake_output), fake_output)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)

discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,

                                 discriminator_optimizer=discriminator_optimizer,

                                 generator=generator,

                                 discriminator=discriminator)
EPOCHS = 75

noise_dim = 100

num_examples_to_generate = 6000

seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function

def train_step(images):

    noise = tf.random.normal([BATCH_SIZE, noise_dim])



    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

      generated_images = generator(noise, training=True)



      real_output = discriminator(images, training=True)

      fake_output = discriminator(generated_images, training=True)



      gen_loss = generator_loss(fake_output)

      disc_loss = discriminator_loss(real_output, fake_output)



    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)



    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
def train(dataset, epochs):

  for epoch in range(epochs):

    start = time.time()



    for image_batch in dataset:

      train_step(image_batch)



    # Produce images for the GIF as we go

    display.clear_output(wait=True)

    generate_and_save_images(generator,

                             epoch + 1,

                             seed)



    # Save the model every 15 epochs

    if (epoch + 1) % 15 == 0:

      checkpoint.save(file_prefix = checkpoint_prefix)



    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))



  # Generate after the final epoch

  display.clear_output(wait=True)

  new_images=generate_and_save_images(generator,

                           epochs,

                           seed)

  return new_images
def generate_and_save_images(model, epoch, test_input):

  # Notice `training` is set to False.

  # This is so all layers run in inference mode (batchnorm).

    new_images = model(test_input, training=False)

    return new_images

    #print(new_images[1].shape)

    #new_images=np.asarray(new_images)

    #new_images=np.reshape(16,28,28)

   # plt.imshow(new_images[1].reshape(28,28))

'''

  fig = plt.figure(figsize=(4,4))



  for i in range(predictions.shape[0]):

    img = image.array_to_img(predictions[i] * 255., scale=False)

    plt.figure()

    plt.imshow(img)

    plt.show()

  '''      

       # plt.subplot(4, 4, i+1)

     # plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')

     # plt.axis('off')



 # plt.savefig('image_at_epoch_{:01d}.png'.format(epoch))

 # plt.show()


%%time



train_dataset = tf.data.Dataset.from_tensor_slices(X_label0).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

gen_images0=train(train_dataset, EPOCHS)



train_dataset = tf.data.Dataset.from_tensor_slices(X_label1).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

gen_images1=train(train_dataset, EPOCHS)



train_dataset = tf.data.Dataset.from_tensor_slices(X_label2).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

gen_images2=train(train_dataset, EPOCHS)



train_dataset = tf.data.Dataset.from_tensor_slices(X_label3).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

gen_images3=train(train_dataset, EPOCHS)



train_dataset = tf.data.Dataset.from_tensor_slices(X_label4).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

gen_images4=train(train_dataset, EPOCHS)



train_dataset = tf.data.Dataset.from_tensor_slices(X_label5).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

gen_images5=train(train_dataset, EPOCHS)



train_dataset = tf.data.Dataset.from_tensor_slices(X_label6).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

gen_images6=train(train_dataset, EPOCHS)



train_dataset = tf.data.Dataset.from_tensor_slices(X_label7).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

gen_images7=train(train_dataset, EPOCHS)



train_dataset = tf.data.Dataset.from_tensor_slices(X_label8).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

gen_images8=train(train_dataset, EPOCHS)



train_dataset = tf.data.Dataset.from_tensor_slices(X_label9).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

gen_images9=train(train_dataset, EPOCHS)

















'''

print(gen_images5.shape)

gen_images5=np.asarray(gen_images5)

gen_images5=gen_images5.reshape(-1,28,28)

print(gen_images5.shape)

plt.imshow(gen_images5[5009])

'''

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


#print(gen_images.shape)

gen_images0=np.asarray(gen_images0)

gen_images0=gen_images0.reshape(-1,28,28,1)

print(gen_images0.shape)

#plt.imshow(gen_images[3])



gen_images1=np.asarray(gen_images1)

gen_images1=gen_images1.reshape(-1,28,28,1)

print(gen_images1.shape)



gen_images2=np.asarray(gen_images2)

gen_images2=gen_images2.reshape(-1,28,28,1)

print(gen_images2.shape)



gen_images3=np.asarray(gen_images3)

gen_images3=gen_images3.reshape(-1,28,28,1)

print(gen_images3.shape)



gen_images4=np.asarray(gen_images4)

gen_images4=gen_images4.reshape(-1,28,28,1)

print(gen_images4.shape)



gen_images5=np.asarray(gen_images5)

gen_images5=gen_images5.reshape(-1,28,28,1)

print(gen_images5.shape)



gen_images6=np.asarray(gen_images0)

gen_images6=gen_images6.reshape(-1,28,28,1)

print(gen_images6.shape)



gen_images7=np.asarray(gen_images7)

gen_images7=gen_images7.reshape(-1,28,28,1)

print(gen_images7.shape)



gen_images8=np.asarray(gen_images8)

gen_images8=gen_images8.reshape(-1,28,28,1)

print(gen_images8.shape)



gen_images9=np.asarray(gen_images9)

gen_images9=gen_images9.reshape(-1,28,28,1)

print(gen_images0.shape)
gen_images=np.vstack((gen_images0,gen_images1,gen_images2,gen_images3,gen_images4,gen_images5,gen_images6,gen_images7,gen_images8,gen_images9))

print(gen_images.shape)

Y_labels=np.vstack((Y_label0,Y_label1,Y_label2,Y_label3,Y_label4,Y_label5,Y_label6,Y_label7,Y_label8,Y_label9,))

Y_labels=Y_labels.reshape(60000)

#print(Y_labels[40000])

#gen_images=gen_images.reshape(-1,28,28)

#plt.imshow(gen_images[48000])
train_CNN_X=np.vstack((train_CNN_X.reshape(-1,28,28,1),gen_images.reshape(-1,28,28,1)))

print(train_CNN_X.shape)

train_CNN_Y=np.vstack((train_CNN_Y,Y_labels))

train_CNN_Y=train_CNN_Y.reshape(120000)

print(train_CNN_Y.shape)
model = keras.Sequential([

    keras.layers.Conv2D(filters=32, kernel_size=(5, 5),padding='Same',activation='relu',input_shape=(28,28,1)),

    keras.layers.Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',),

    keras.layers.BatchNormalization(momentum=.15),

    keras.layers.MaxPool2D(pool_size=(2,2)),

    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'),

    keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'),

    keras.layers.BatchNormalization(momentum=0.15),

    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)),

    keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'),

    keras.layers.BatchNormalization(momentum=.15),

    keras.layers.MaxPool2D(pool_size=(2,2)),

    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.Dropout(0.4),

    keras.layers.Dense(10, activation='softmax')

])

model.summary()
optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer=optimizer,

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
#train_CNN_X=train_CNN_X.reshape(-1,28,28,1)



model.fit(train_CNN_X, train_CNN_Y, epochs=75)



#test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

#print('\nTest accuracy:', test_acc)


#prediction=model.predict(gen_images)


#labels=np.argmax(prediction,axis=1)



#print(labels)
#model.fit(gen_images, labels, epochs=20)

test1=test.reshape(-1,28,28,1)

prediction1=model.predict(test1)

prediction1=np.argmax(prediction1,axis=1)

#print(prediction1)

submission_sample['label']=prediction1

#print(submission_sample.head(10))

submission_sample.to_csv('submission.csv',index=False)
