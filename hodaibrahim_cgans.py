import os
import matplotlib.pyplot as plt
# Import data
faces ='../input/utkface-new/UTKFace'
os.listdir('../input/utkface-new')
from PIL import Image

im =Image.open(faces+'/100_1_0_20170112215032192.jpg.chip.jpg').resize((128,128))
im
len(os.listdir(faces))
'100_1_0_20170112215032192.jpg.chip.jpg'.split('_')[1]
all_faces = os.listdir(faces)
all_faces[:10]
labels = []
for face_name in all_faces:
    i = int(face_name.split('_')[1])
    labels.append(i)
len(labels)

import numpy as np
import pandas as pd 
filenames = all_faces
categories = []
for filename in filenames:
    category = int(filename.split('_')[1])
    if category == 1 :
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
df.head()
df.tail()
df['category'].value_counts().plot.bar()
import imageio
import cv2

X_data =[]
for file in all_faces:
    face = imageio.imread(faces+'/'+file)
    face = cv2.resize(face, (32, 32) )
    X_data.append(face)
X = np.squeeze(X_data)
X.shape
# normalize data
X = X.astype('float32')
X = X / 255
from keras.utils import to_categorical
categorical_labels = to_categorical(labels, num_classes=2)
categorical_labels[:10]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, categorical_labels, test_size=0.33, random_state=42)
len(X_train)
from keras.layers import Input, Dense, Flatten, Dropout, Reshape, Concatenate
from keras.layers import BatchNormalization, Activation, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.models import Model
from keras.optimizers import Adam
import sys
def get_generator(input_layer, condition_layer):

    inp = Concatenate()([input_layer, condition_layer])
  
    x = Dense(128 * 8 * 8, activation='relu')(inp)    
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Reshape((8, 8, 128))(x)

    x = Conv2D(128, kernel_size=4, strides=1,padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)    
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(128, kernel_size=5, strides=1,padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)    
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(128, kernel_size=5, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(128, kernel_size=5, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU(alpha=0.1)(x)
                      
    x = Conv2D(3, kernel_size=5, strides=1, padding="same")(x)
    out = Activation("tanh")(x)

    model = Model(inputs=[input_layer, condition_layer], outputs=out)
    model.summary()
  
    return model, out
def get_discriminator(input_layer, condition_layer):
    hid = Conv2D(128, kernel_size=3, strides=1, padding='same')(input_layer)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    hid = Flatten()(hid)
  
    merged_layer = Concatenate()([hid, condition_layer])
    hid = Dense(512, activation='relu')(merged_layer)
    #hid = Dropout(0.4)(hid)
    out = Dense(1, activation='sigmoid')(hid)

    model = Model(inputs=[input_layer, condition_layer], outputs=out)

    model.summary()

    return model, out
def generate_random_labels(n):
    y = np.random.choice(10, n)
    return y

generate_random_labels(2)

from keras.preprocessing import image


def one_hot_encode(y):
    z = np.zeros((len(y), 2))
    idx = np.arange(len(y))
    z[idx, y] = 1
    return z


def generate_noise(n_samples, noise_dim):
    X = np.random.normal(0, 1, size=(n_samples, noise_dim))
    return X

def generate_random_labels(n):
    y = np.random.choice(2, n)
    y = one_hot_encode(y)
    return y

tags = ['Male', 'Female']
  
def show_samples(batchidx):
    samples = 2
    z = np.random.normal(loc=0, scale=1, size=(samples, 100))
    labels = to_categorical(np.arange(0, 2).reshape(-1, 1), num_classes=2)
        
    x_fake = generator.predict([z, labels])
    x_fake = np.clip(x_fake, -1, 1)
    x_fake = (x_fake + 1) * 127
    x_fake = np.round(x_fake).astype('uint8')

    for k in range(samples):
        plt.subplot(2, 5, k + 1, xticks=[], yticks=[])
        plt.imshow(x_fake[k])
        plt.title(tags[k])

    plt.tight_layout()
    plt.show()
# GAN creation
img_input = Input(shape=(32,32,3))
disc_condition_input = Input(shape=(2,))

discriminator, disc_out = get_discriminator(img_input, disc_condition_input)
discriminator.compile(optimizer=Adam(0.001, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False

noise_input = Input(shape=(100,))
gen_condition_input = Input(shape=(2,))
generator, gen_out = get_generator(noise_input, gen_condition_input)

gan_input = Input(shape=(100,))
x = generator([gan_input, gen_condition_input])
gan_out = discriminator([x, disc_condition_input])
gan = Model([gan_input, gen_condition_input, disc_condition_input], gan_out)
gan.summary()

gan.compile(optimizer=Adam(0.001, 0.5), loss='binary_crossentropy')
BATCH_SIZE = 16

print ("Training shape: {}, {}".format(X_train.shape, y_train.shape))
 
num_batches = int(X_train.shape[0]/BATCH_SIZE)

print ("Num batches: {}".format(num_batches))
# Array to store samples for experience replay
exp_replay = []
N_EPOCHS = 20
for epoch in range(N_EPOCHS):

    cum_d_loss = 0.
    cum_g_loss = 0.

    for batch_idx in range(num_batches):
        # Get the next set of real images to be used in this iteration
        images = X_train[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]
        labels = y_train[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]

        noise_data = generate_noise(BATCH_SIZE, 100)
        random_labels = generate_random_labels(BATCH_SIZE)
        generated_images = generator.predict([noise_data, labels])

        noise_prop = 0.1 

        # Prepare labels for real data
        true_labels = np.zeros((BATCH_SIZE, 1)) + np.random.uniform(low=0.0, high=0.1, size=(BATCH_SIZE, 1))
        flipped_idx = np.random.choice(np.arange(len(true_labels)), size=int(noise_prop*len(true_labels)))
        true_labels[flipped_idx] = 1 - true_labels[flipped_idx]

        d_loss_true = discriminator.train_on_batch([images, labels], true_labels)

        gene_labels = np.ones((BATCH_SIZE, 1)) - np.random.uniform(low=0.0, high=0.1, size=(BATCH_SIZE, 1))
        flipped_idx = np.random.choice(np.arange(len(gene_labels)), size=int(noise_prop*len(gene_labels)))
        gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]

        d_loss_gene = discriminator.train_on_batch([generated_images, labels], gene_labels)

        r_idx = np.random.randint(BATCH_SIZE)
        exp_replay.append([generated_images[r_idx], labels[r_idx], gene_labels[r_idx]])

        # If we have enough points, do experience replay
        if len(exp_replay) == BATCH_SIZE:
            generated_images = np.array([p[0] for p in exp_replay])
            labels = np.array([p[1] for p in exp_replay])
            gene_labels = np.array([p[2] for p in exp_replay])
            expprep_loss_gene = discriminator.train_on_batch([generated_images, labels], gene_labels)
            exp_replay = []

        d_loss = 0.5 * np.add(d_loss_true, d_loss_gene)
        cum_d_loss += d_loss

        # Train generator
        noise_data = generate_noise(BATCH_SIZE, 100)
        random_labels = generate_random_labels(BATCH_SIZE)
        g_loss = gan.train_on_batch([noise_data, random_labels, random_labels], np.zeros((BATCH_SIZE, 1)))
        cum_g_loss += g_loss

        if batch_idx % 500 == 0:
            print(batch_idx, d_loss_true[0], d_loss_gene[0], g_loss, "Cumu:", cum_d_loss[0]/(batch_idx+1), cum_g_loss/(batch_idx+1))

    print('\tEpoch: {}, Generator Loss: {}, Discriminator Loss: {}'.format(epoch+1, cum_g_loss/num_batches, cum_d_loss/num_batches))
    show_samples("epoch" + str(epoch))



