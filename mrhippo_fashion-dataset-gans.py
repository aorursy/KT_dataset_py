import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objs as go #visualization
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True) 
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")
#test = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")
train.head()
train.info()
# Keras(deep learning library)
from keras.layers import Dense, Dropout, Input, ReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

y_data = train["label"].values
x_data = train.drop(["label"],axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size = 0.2, random_state = 42)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
x_train = (x_train.astype(np.float32)-127.5)/127.5

print(x_train.shape)
print(x_test.shape)
fig, axs = plt.subplots(2, 3, figsize=(13, 8), sharey=True)

axs[0,0].imshow(x_test.reshape(12000,28,28)[1])

axs[0,1].imshow(x_test.reshape(12000,28,28)[2])

axs[0,2].imshow(x_test.reshape(12000,28,28)[11])

axs[1,0].imshow(x_test.reshape(12000,28,28)[101])

axs[1,1].imshow(x_test.reshape(12000,28,28)[7000])

axs[1,2].imshow(x_test.reshape(12000,28,28)[11000])
fig.suptitle("Dataset Samples",fontsize=16)

plt.show()
def create_generator():
    
    generator = Sequential()
    generator.add(Dense(units = 512, input_dim = 100))
    generator.add(ReLU())
    
    generator.add(Dense(units = 512))
    generator.add(ReLU())
    
    generator.add(Dense(units = 1024))
    generator.add(ReLU())
    
    generator.add(Dense(units = 1024))
    generator.add(ReLU())
    
    generator.add(Dense(units = 784))
    
    generator.compile(loss ="binary_crossentropy",
                     optimizer = Adam(lr = 0.0001, beta_1 = 0.5))
    
    return generator

g = create_generator()
g.summary()
def create_discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(units = 1024,input_dim = 784)) 
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.4))
    
    discriminator.add(Dense(units = 512)) 
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.4))
    
    discriminator.add(Dense(units = 256)) 
    discriminator.add(ReLU())
    
    discriminator.add(Dense(units = 1, activation = "sigmoid"))
    
    discriminator.compile(loss = "binary_crossentropy",
                         optimizer = Adam(lr = 0.0001, beta_1 = 0.5))
    return discriminator

d = create_discriminator()
d.summary()
def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs = gan_input, outputs = gan_output)
    gan.compile(loss = "binary_crossentropy", optimizer = "adam")
    return gan

gan = create_gan(d,g)
gan.summary()
import time

D_loss = []
G_loss = []
epochs = 1000 
batch_size = 256
current_time = time.time()

for e in range(epochs):
    start_time = time.time()
    for _ in range(batch_size):
        # I reccomend you to look "Training Diagram" (at the top) 
        noise = np.random.normal(0,1, [batch_size,100])
        
        generated_images = g.predict(noise)
       
        image_batch = x_train[np.random.randint(low = 0, high = x_train.shape[0], size = batch_size)] #get samples from real data
        
        x = np.concatenate([image_batch, generated_images])
        
        y_dis = np.zeros(batch_size*2) 
        y_dis[:batch_size] = 1 # we labeled real images as 1 and generated images as 0
        
        d.trainable = True
        d_loss = d.train_on_batch(x,y_dis) # we are training discriminator (train_on_batch)
        
        noise = np.random.normal(0,1,[batch_size,100])
        
        y_gen = np.ones(batch_size) # our generator says "these images are real"
        
        d.trainable = False
        
        g_loss = gan.train_on_batch(noise, y_gen) #train_on_batch
        
        D_loss.append(d_loss)
        G_loss.append(g_loss)
        
    if (e%20 == 0) or (e == epochs-1) :
        print("epochs: ",e)
    if e == epochs-1:
        print("Time since start: {}".format(np.round(start_time - current_time)))
        print("Training Complete.")
    
    # printing results
    if e%100 == 0:
        print("Time since start: {}".format(np.round(start_time - current_time)))
        
        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = g.predict(noise)
        generated_images = generated_images.reshape(100,28,28)

        fig, axs = plt.subplots(3, 4, figsize=(13, 8), sharey=True)
        axs[0,0].imshow(generated_images[66], interpolation = "nearest")
        axs[0,0].axis("off")

        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = g.predict(noise)
        generated_images = generated_images.reshape(100,28,28)
        axs[0,1].imshow(generated_images[66], interpolation = "nearest")
        axs[0,1].axis("off")

        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = g.predict(noise)
        generated_images = generated_images.reshape(100,28,28)
        axs[0,2].imshow(generated_images[66], interpolation = "nearest")
        axs[0,2].axis("off")
        
        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = g.predict(noise)
        generated_images = generated_images.reshape(100,28,28)
        axs[0,3].imshow(generated_images[66], interpolation = "nearest")
        axs[0,3].axis("off")

        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = g.predict(noise)
        generated_images = generated_images.reshape(100,28,28)
        axs[1,0].imshow(generated_images[66], interpolation = "nearest")
        axs[1,0].axis("off")

        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = g.predict(noise)
        generated_images = generated_images.reshape(100,28,28)
        axs[1,1].imshow(generated_images[66], interpolation = "nearest")
        axs[1,1].axis("off")

        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = g.predict(noise)
        generated_images = generated_images.reshape(100,28,28)
        axs[1,2].imshow(generated_images[66], interpolation = "nearest")
        axs[1,2].axis("off")
        
        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = g.predict(noise)
        generated_images = generated_images.reshape(100,28,28)
        axs[1,3].imshow(generated_images[66], interpolation = "nearest")
        axs[1,3].axis("off")

        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = g.predict(noise)
        generated_images = generated_images.reshape(100,28,28)
        axs[2,0].imshow(generated_images[66], interpolation = "nearest")
        axs[2,0].axis("off")

        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = g.predict(noise)
        generated_images = generated_images.reshape(100,28,28)
        axs[2,1].imshow(generated_images[66], interpolation = "nearest")
        axs[2,1].axis("off")

        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = g.predict(noise)
        generated_images = generated_images.reshape(100,28,28)
        axs[2,2].imshow(generated_images[66], interpolation = "nearest")
        axs[2,2].axis("off")
        
        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = g.predict(noise)
        generated_images = generated_images.reshape(100,28,28)
        axs[2,3].imshow(generated_images[66], interpolation = "nearest")
        axs[2,3].axis("off")
        
        fig.suptitle("{} Epochs Result".format(str(e)),fontsize=16)
        plt.show()   
#from keras import models
g.save("generator_1000epcohs.h5")
#generator_save = models.load_model("generator_1000epcohs.h5")
#£generator_save.summary()
plt.figure(figsize = (13,8))
plt.plot(noise)
plt.title("Noise")
plt.show()
fig = plt.figure(figsize = (13,5))
plt.plot(D_loss, label = "Discriminator Loss")
plt.plot(G_loss, label = "Generator Loss")
plt.legend()
plt.xlabel("Epochs and Batches (epochs*batch size)")
plt.ylabel("Loss")
plt.title("Discriminator and Generator Losses")

plt.show()
noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)

fig, axs = plt.subplots(3, 5, figsize=(13, 8), sharey=True)
axs[0,0].imshow(generated_images[66], interpolation = "nearest")
axs[0,0].axis("off")

noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
axs[0,1].imshow(generated_images[66], interpolation = "nearest")
axs[0,1].axis("off")

noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
axs[0,2].imshow(generated_images[66], interpolation = "nearest")
axs[0,2].axis("off")

noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
axs[0,3].imshow(generated_images[66], interpolation = "nearest")
axs[0,3].axis("off")

noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
axs[0,4].imshow(generated_images[66], interpolation = "nearest")
axs[0,4].axis("off")

noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
axs[1,0].imshow(generated_images[66], interpolation = "nearest")
axs[1,0].axis("off")

noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
axs[1,1].imshow(generated_images[66], interpolation = "nearest")
axs[1,1].axis("off")

noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
axs[1,2].imshow(generated_images[66], interpolation = "nearest")
axs[1,2].axis("off")

noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
axs[1,3].imshow(generated_images[66], interpolation = "nearest")
axs[1,3].axis("off")

noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
axs[1,4].imshow(generated_images[66], interpolation = "nearest")
axs[1,4].axis("off")

noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
axs[2,0].imshow(generated_images[66], interpolation = "nearest")
axs[2,0].axis("off")

noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
axs[2,1].imshow(generated_images[66], interpolation = "nearest")
axs[2,1].axis("off")

noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
axs[2,2].imshow(generated_images[66], interpolation = "nearest")
axs[2,2].axis("off")

noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
axs[2,3].imshow(generated_images[66], interpolation = "nearest")
axs[2,3].axis("off")

noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
axs[2,4].imshow(generated_images[66], interpolation = "nearest")
axs[2,4].axis("off")

fig.suptitle("{} Epochs Result".format(str(epochs)),fontsize=16)

plt.show()