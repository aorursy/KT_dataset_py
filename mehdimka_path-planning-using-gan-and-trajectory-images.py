import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.layers import Input, Dense, Reshape, Flatten, Dropout

from keras.layers import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU

from keras.models import Sequential, Model

from keras.optimizers import Adam



from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder



import os

import time

from pylab import *
data_path = "../input/dataset/dataset"
sample_data = os.path.join(data_path, "00_010.csv")

sample_df = pd.read_csv(sample_data, index_col=None, header=None)

sample_df
%matplotlib inline

from matplotlib import pyplot as plt

plt.imshow(sample_df.values, interpolation='nearest')

plt.axis('off')

plt.show()
def loadData(path=data_path):

    files = os.listdir(path)

    data = []

    labels = []

    for fn in files:

        ffn = os.path.join(path, fn)

        df = pd.read_csv(ffn, index_col=None, header=None)

        df[df==2]=0

        data.append(df.values)

        label = int(fn[0:2])

        labels.append(label)

    data = np.array(data) 

    return data, labels
data, labels = loadData() 
print(labels)
defined_path_classes = pd.read_csv("../input/pathClasses.csv", index_col=None)

defined_path_classes
class GAN():

    def __init__(self, img_rows=19, img_cols=13, channels=1):

        self.img_rows = img_rows

        self.img_cols = img_cols

        self.channels = channels

        self.img_shape = (self.img_rows, self.img_cols, self.channels)



        optimizer = Adam(0.0002, 0.5)



        # Build and compile the discriminator

        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',

                                   optimizer=optimizer,

                                   metrics=['accuracy'])



        # Build and compile the generator

        self.generator = self.build_generator()

        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)



        # The generator takes noise as input and generated imgs

        z = Input(shape=(100,))

        img = self.generator(z)



        # For the combined model we will only train the generator

        self.discriminator.trainable = False



        # The valid takes generated images as input and determines validity

        valid = self.discriminator(img)



        # The combined model  (stacked generator and discriminator) takes

        # noise as input => generates images => determines validity 

        self.combined = Model(z, valid)

        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)



    def build_generator(self):



        noise_shape = (100,)



        model = Sequential()



        model.add(Dense(256, input_shape=noise_shape))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))

        model.add(Reshape(self.img_shape))



        model.summary()



        noise = Input(shape=noise_shape)

        img = model(noise)



        return Model(noise, img)



    def build_discriminator(self):



        img_shape = (self.img_rows, self.img_cols, self.channels)



        model = Sequential()



        model.add(Flatten(input_shape=img_shape))

        model.add(Dense(512))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))

        model.summary()



        img = Input(shape=img_shape)

        validity = model(img)



        return Model(img, validity)



    def train(self, X_train, y_, epochs=1000, batch_size=2):



        X_train = 2 * (X_train.astype(np.float32)) - 1

        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)



        for epoch in range(epochs):



            # ---------------------

            #  Train Discriminator

            # ---------------------



            # Select a random half batch of images

            idx = np.random.randint(0, X_train.shape[0], half_batch)

            imgs = X_train[idx]



            noise = np.random.normal(0, 1, (half_batch, 100))



            # Generate a half batch of new images

            gen_imgs = self.generator.predict(noise)



            # Train the discriminator

            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))

            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)



            # ---------------------

            #  Train Generator

            # ---------------------



            noise = np.random.normal(0, 1, (batch_size, 100))



            # The generator wants the discriminator to label the generated samples as valid (ones)

            valid_y = np.array([1] * batch_size)



            # Train the generator

            g_loss = self.combined.train_on_batch(noise, valid_y)
gan = GAN()

start_time = time.time()

gan.train(data, labels, epochs=10000, batch_size=8)

print("total training time: %s seconds ---" % (time.time() - start_time))
gen_samples_row, gen_samples_col = 3,4

count = gen_samples_row*gen_samples_col

noise = np.random.normal(0, 1, (count, 100))

print(noise.shape)
noise[0]
# Generate images from noise data

gen_imgs = gan.generator.predict(noise)

print(gen_imgs.shape)
gen_imgs = 0.5 * gen_imgs + 0.5
gen_imgs[0].shape
fig, axs = plt.subplots(gen_samples_row, gen_samples_col, figsize=(4.5,5))

cnt = 0

fig.subplots_adjust(hspace=0.5)

for i in range(gen_samples_row):

    for j in range(gen_samples_col):

        axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap=plt.cm.gray)

        axs[i,j].axis('off')

    autoAxis = axs[i,j].axis()

    rec = Rectangle((autoAxis[0]-0.1,autoAxis[2]-0.2),(autoAxis[1]-autoAxis[0])+.2,(autoAxis[3]-autoAxis[2])+0.1,fill=False, lw=0.5)

    rec = axs[i,j].add_patch(rec)

    rec.set_clip_on(False)

    cnt += 1

plt.show()

plt.close()
class PathClassifier:

    

    def __init__(self, num_pixels=13*19, num_classes=6):

        self.model = self.build_classifier_base(num_pixels, num_classes)



    def build_classifier_base(self, num_pixels, num_classes):

        # fix random seed for reproducibility

        seed = 7

        np.random.seed(seed)

        model = Sequential()

        model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))

        model.add(Dropout(0.2))

        model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

        # Compile model

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model



    def train_model(self, X_train, y_train, eps=10, batch_size=20):

        num_classes = y_train.shape[1]

        num_pixels = X_train.shape[1]        

        self.model.fit(X_train, y_train, epochs=eps, batch_size=batch_size, verbose=2)

        return self.model



    def evaluate(self, X_test, y_test):        

        scores = self.model.evaluate(X_test, y_test, verbose=0)

        accuracy = scores[1] * 100

        print("Classification accuracy: %.2f%%" % accuracy)

        return accuracy
path_classifier = PathClassifier()

lblEnc = LabelEncoder()

labels = lblEnc.fit_transform(labels)



num_pixels = data.shape[1] * data.shape[2]

data = data.reshape(data.shape[0], num_pixels).astype('float32')

labels = np_utils.to_categorical(labels)



num_classes = labels.shape[1]



lnx = int(len(data) * 0.7)

X_train, X_test = data[:lnx], data[lnx:]

y_train, y_test = labels[:lnx], labels[lnx:]



path_classifier.train_model(X_train, y_train, eps=10, batch_size=i*10)

path_classifier.evaluate(X_test, y_test)
gen_imgs2 = gen_imgs.reshape(gen_imgs.shape[0], gen_imgs.shape[1]*gen_imgs.shape[2]).astype('float32')

classes = path_classifier.model.predict(gen_imgs2)

classes = np.argmax(classes, axis=1)
fig, axs = plt.subplots(gen_samples_row, gen_samples_col, figsize=(4.5,5))

cnt = 0

fig.subplots_adjust(hspace=0.5)

for i in range(gen_samples_row):

    for j in range(gen_samples_col):

        axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap=plt.cm.gray)

        axs[i,j].set_title('class ' + str(classes[cnt]))

        axs[i,j].axis('off')

    autoAxis = axs[i,j].axis()

    rec = Rectangle((autoAxis[0]-0.1,autoAxis[2]-0.2),(autoAxis[1]-autoAxis[0])+.2,(autoAxis[3]-autoAxis[2])+0.1,fill=False, lw=0.5)

    rec = axs[i,j].add_patch(rec)

    rec.set_clip_on(False)

    cnt += 1

plt.show()

plt.close()