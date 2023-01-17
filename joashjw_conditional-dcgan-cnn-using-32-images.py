import pandas as pd

import numpy as np

from keras.models import Model

from keras.preprocessing.image import array_to_img

from keras.layers import (Input, Dense, Dropout, Flatten, Reshape, Conv2D, UpSampling2D, Embedding,

                          BatchNormalization, Flatten, LeakyReLU, GaussianNoise, Multiply)

from keras.optimizers import Adam, SGD

from keras.initializers import TruncatedNormal

import matplotlib.pyplot as plt
train_data = pd.read_csv("../input/digit-recognizer/train.csv")

train_data.label = train_data.label.astype('category')
img_size = 28

b_size = 32

channels = 1

num_of_classes = train_data.label.nunique()

latent_dim = 100

np.random.seed(2019)



def getImages(label, size):

    indexes = np.random.choice(train_data[train_data.label==label].index, size)

    rows = train_data.drop(columns=['label']).loc[indexes]

    images = np.reshape(np.array(rows), newshape=(size,img_size,img_size,channels)) # shape in (size, width, height, channels)

    return np.array(images)
all_data = []

for i in range(10):

    all_data.append((getImages(i, 320+32)-127.5)/127.5) # scale to [-1, 1]
noise_input = Input(shape=(latent_dim,))

c_input = Input(shape=(1,))



embedding = Flatten()(Embedding(num_of_classes+1, num_of_classes*10)(c_input))

merge_layer = Multiply()([noise_input, embedding])

 

x = Dense(7*7*256)(merge_layer)

x = BatchNormalization()(x)

x = LeakyReLU(0.2)(x)

x = Reshape((7,7,256))(x)

x = UpSampling2D()(x)



# 14x14x128

x = Conv2D(128, 5, padding='same')(x)

x = LeakyReLU(0.2)(x)

x = BatchNormalization()(x)

x = UpSampling2D()(x)



# 28x28x64

x = Conv2D(64, 3, padding='same')(x)

x = LeakyReLU(0.2)(x)

x = BatchNormalization()(x)



generator_output = Conv2D(channels, 3, activation='tanh', padding='same')(x)

generator = Model(inputs=[noise_input, c_input], outputs=generator_output)

generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, decay=1e-6))

generator.summary()
# initial generated image

noise = np.random.normal(0,1,size=latent_dim)

noise = np.reshape(noise, newshape=(1,latent_dim))

condition = np.array([0])

image = generator.predict([noise, condition])

image = np.reshape(image, newshape=(img_size,img_size,channels))

image = array_to_img(image)

image.resize((128,128))
# simple cnn model

discriminator_input = Input(shape=(img_size,img_size,channels))



x = GaussianNoise(0.2)(discriminator_input)

x = Conv2D(64, 5, padding='same', kernel_initializer=TruncatedNormal(stddev=0.02))(x)

x = BatchNormalization(momentum=0.5)(x)

x = LeakyReLU(0.2)(x)



x = GaussianNoise(0.2)(x)

x = Conv2D(128, 5, padding='same', kernel_initializer=TruncatedNormal(stddev=0.02))(x)

x = BatchNormalization(momentum=0.5)(x)

x = LeakyReLU(0.2)(x)



x = Flatten()(x)

x = Dense(1000)(x)

x = LeakyReLU(0.2)(x)

x = Dropout(0.3)(x)

embedding_d = Flatten()(Embedding(num_of_classes+1, num_of_classes*100)(c_input))

discriminator_merge = Multiply()([x, embedding_d])

x = Dense(128)(discriminator_merge)

x = LeakyReLU(0.2)(x)

x = Dropout(0.3)(x)

discriminator_output = Dense(1, activation="sigmoid")(x)



discriminator = Model(inputs=[discriminator_input, c_input], outputs=discriminator_output)

discriminator.trainable = True

discriminator.compile(loss='binary_crossentropy', optimizer=SGD(lr=1e-3,decay=1e-6, momentum=0.9, nesterov=True))

discriminator.summary()
def getConditions(condition):

    conditions = np.array([condition])

    return np.reshape(np.array([conditions for i in range(b_size)]), newshape=(b_size, 1))



conditions = np.array([getConditions(i) for i in range(num_of_classes)])
for epoch in range(100):

    i = int(epoch%10)

    d_loss_real = discriminator.train_on_batch([all_data[i][:32], conditions[i]], np.ones(b_size))

    fakes = generator.predict([np.random.normal(0,1,size=(b_size, latent_dim)), conditions[i]])

    d_loss_fake = discriminator.train_on_batch([fakes, conditions[i]], np.array([0.1 for i in range(b_size)]))

    if epoch%20==0:

        print("Real Loss:", d_loss_real)

        print("Fake Loss:", d_loss_fake)
gan_output = discriminator([generator_output, c_input])

gan = Model(inputs=[noise_input, c_input], outputs=gan_output)

discriminator.trainable = False

gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, decay=1e-6))
def getFakeData(iteration):

    noise = np.random.normal(0,1,size=(b_size,latent_dim))

    condition = conditions[iteration]

    x_fake = generator.predict([noise, condition])

    y_fake = np.array([0.1 for i in range(len(x_fake))]) #0.1 to represent fake

    return x_fake, y_fake



def getNoisyData(y_real): # occasionally flip the labels when training the discriminator

    if np.random.random_integers(0,1) == 0:

        return y_real

    else:

        max_flips = len(y_real)//8

        number_of_flips = np.random.randint(0,max_flips)

        for i in np.random.choice([i for i in range(max_flips)], number_of_flips):

            y_real[i] = 0.1

        return y_real

    

def train_discriminator(iteration):

    x_real, y_real = getRealData(iteration)

    x_fake, y_fake = getFakeData(iteration)

    

    # train discriminator

    discriminator.trainable = True

    y_real = getNoisyData(y_real)

    condition = conditions[iteration]

    loss_real = discriminator.train_on_batch([x_real, condition], y_real)

    loss_fake = discriminator.train_on_batch([x_fake, condition], y_fake)

    return [loss_real, loss_fake]

    

def train_gan(iteration):

    # train generator

    discriminator.trainable = False

    noise = np.random.normal(0,1,size=(b_size,latent_dim))

    condition = conditions[iteration]

    y_gen = np.ones(len(noise))

    g_loss = gan.train_on_batch([noise, condition], y_gen)

    return g_loss



def getRealData(iteration):

    x_real = all_data[iteration][:32]

    y_real = np.array([(0.8+0.01*iteration) for i in range(len(x_real))]) # label smoothing

    return x_real, y_real
def plotImage(condition):

    noise = np.random.normal(0,1,size=latent_dim)

    noise = np.reshape(noise, newshape=(1,latent_dim))

    image = generator.predict([noise, conditions[condition]])

    image = np.reshape(image, newshape=(img_size,img_size,channels))

    image = array_to_img(image)

    return image
epoch = 1

while (epoch<=4000):

    for i in range(10):

        # train 2x discriminator then 1x generator

        #train_discriminator(i)

        loss_real, loss_fake = train_discriminator(i)

        g_loss = train_gan(i)

    if epoch%500 == 0:

        print("Epoch ", epoch)

        print("Discriminator Loss (Real):", loss_real)

        print("Discriminator Loss (Fake):", loss_fake)

        d_loss = 0.5*(loss_real+loss_fake)

        print("Discriminator Loss (Total):", d_loss)

        print("Gen Loss:", g_loss)

        fig=plt.figure(figsize=(16, 16))

        for i in range(1, 11):

            img = plotImage(i-1)

            fig.add_subplot(1, 10, i)

            plt.imshow(img)

        plt.show()

    epoch+=1
fig=plt.figure(figsize=(16, 16))

for i in range(1, 11):

    img = plotImage(i-1)

    fig.add_subplot(1, 10, i)

    plt.imshow(img)

plt.show()
discriminator.save('discriminator.h5')

generator.save('generator.h5')

gan.save('gan.h5')
#from keras.models import load_model

#generator = load_model('../input/mnist-generator/generator.h5') # generator from commit 4
def createCNN():

    cnn_input = Input(shape=(img_size,img_size,channels))



    x = Conv2D(32, 3, padding='same', activation='relu')(cnn_input)

    x = BatchNormalization()(x)

    x = Conv2D(32, 3, padding='same', activation='relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(32, 5, padding='same', activation='relu', strides=2)(x)

    x = BatchNormalization()(x)

    x = Dropout(0.4)(x)



    x = Conv2D(64, 3, padding='same', activation='relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(64, 3, padding='same', activation='relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(64, 5, padding='same', activation='relu', strides=2)(x)

    x = BatchNormalization()(x)

    x = Dropout(0.4)(x)



    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.4)(x)



    cnn_output = Dense(10, activation='softmax')(x)

    cnn = Model(inputs=cnn_input, outputs=cnn_output)

    cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

    return cnn
# val data of 320 images

val_data = []

val_labels = []



for i in range(10):

    val_data.append(all_data[i][32:])

    onehot = np.array([0 for i in range(num_of_classes)])

    onehot[i] = 1

    val_labels.append(np.array([onehot for y in range(320)]))

    

val_labels = np.reshape(np.array(val_labels), newshape=(320*10, 10))

val_data = np.reshape(np.array(val_data), newshape=(320*10, img_size, img_size, channels))
cnn = createCNN()

labels = []

real_images = []

for i in range(10):

    onehot = np.array([0 for i in range(num_of_classes)])

    onehot[i] = 1

    onehot_labels = np.array([onehot for y in range(32)])

    labels.append(onehot_labels)

    real, useless_var = getRealData(i)

    del useless_var

    real_images.append(real)

labels = np.reshape(np.array(labels), newshape=(320, 10))



real_images = np.reshape(np.array(real_images), newshape=(320, img_size, img_size, channels))

cnn.fit(x=real_images, y=labels, batch_size=b_size, epochs=20)

cnn.evaluate(x=val_data, y=val_labels)
def getNewData(iteration):

    # get one-hot labels for the 3200 images

    onehot = np.array([0 for i in range(num_of_classes)])

    onehot[iteration] = 1

    onehot_labels = np.array([onehot for y in range(3200)])

    

    # conditions for 3200 images

    new_conditions = np.array([np.array([iteration]) for y in range(3200)])



    # generate 3200-32 images

    noise = np.random.normal(0,1,size=(3200-b_size,latent_dim))

    generated = generator.predict([noise, new_conditions[:3200-b_size]])

     # add original 32 real images with generated images

    new_data = np.concatenate((all_data[iteration][:32], generated))

    

    return new_data, onehot_labels
labels = []

new_data = []



for i in range(10):

    temp_data, label = getNewData(i)

    labels.append(label)

    new_data.append(temp_data)

    

labels = np.reshape(np.array(labels), newshape=(3200*10, 10))

new_data = np.reshape(np.array(new_data), newshape=(3200*10, img_size, img_size, channels))
cnn = createCNN()

cnn.fit(x=new_data, y=labels, batch_size=b_size, epochs=20)

cnn.evaluate(x=val_data, y=val_labels)
test_data = pd.read_csv("../input/digit-recognizer/test.csv")

image_id = np.array([i for i in range(1,len(test_data)+1)])

test_data = np.reshape(np.array(test_data)/255, newshape=(len(test_data), img_size, img_size, 1))

predictions = cnn.predict(test_data)
predictions = predictions.argmax(axis=1)

submission = pd.DataFrame({"ImageId":image_id, "Label":predictions})

submission.to_csv("gan_cnn.csv", index=False)