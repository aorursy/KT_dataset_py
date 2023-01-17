#Let's read the data

import numpy as np

import pandas as pd

data = pd.read_csv("../input/iris/Iris.csv")
#How it looks like?

print(type(data))

print(data.head(2))
#Lets convert the dataframe into an array so its easier to manipulate (at least that is what I think)

data_np = data.to_numpy()

print(type(data_np))

print(data_np.shape)
#How many classes of Iris do we have?

Iris = {i for i in data_np[:,-1]}

print(Iris)
#I think we don't need the first column (ID) and the last column is the label so:

print(data_np[:10,0])

labels = data_np[:,-1]

data_np = data_np[:,1:5]

print(data_np.shape)

print(labels[:10])
#Hot encoding time!

def hot_encode(N, class_):

    dic = {}

    for i,j in enumerate(class_):

        dic[j] = i

    array = np.zeros((N,len(class_)))

    for i,j in enumerate(labels):

        array[i,dic[j]] = 1

    return array



print(labels[0])

print(labels[50])

print(labels[120])

label_h = hot_encode(data_np.shape[0], Iris)

print(label_h[0])

print(label_h[50])

print(label_h[120])

print(label_h.shape)
#Ok now lets build our NN

import tensorflow as tf

def build_NN():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(4, activation="relu"))

    model.add(tf.keras.layers.Dense(128, activation="relu"))

    #model.add(tf.keras.layers.Dense(256, activation="relu"))

    model.add(tf.keras.layers.Dense(64, activation="relu"))

    model.add(tf.keras.layers.Dense(3, activation="softmax")) 

    

    return model

model = build_NN()

    
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
from keras.callbacks import EarlyStopping

#label_h = np.asarray(label_h)

data_np = np.asarray(data_np).astype(np.float32)

label_h = np.asarray(label_h).astype(np.float32)



print(data_np.shape)

print(label_h.shape)

print(type(data_np))

print(type(label_h))
model.fit(data_np, label_h, epochs = 100, callbacks = [EarlyStopping(patience=10)])
model.save("Iris_model.h5")
from tqdm import tqdm
import random

#Instead of using the fit function let's use GradientTape to have more control on the traing

learning_rate=[1e-1,1e-2,1e-4,1e-5] #Let's visualize how changing the learning rate value, changes the loss function

Epochs = 100

loss_history = np.zeros((len(learning_rate),Epochs))



for i,rate in enumerate(learning_rate):

    model = build_NN()

    optimizer = tf.keras.optimizers.SGD(learning_rate=rate) # define our optimizer

    batch_size = 10

    x =list(range(Epochs)) 

    #x = list(range(int(data_np.shape[0]/batch_size)))

    if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

    for idx in tqdm(range(0, Epochs)):

        r = random.randint(0,140)

        data_np_b, label_h_b = data_np[r:r+batch_size,:], label_h[r:r+batch_size,:]

        with tf.GradientTape() as tape:

            logits = model(data_np_b) #Performance of the neural network

            loss_value = tf.keras.backend.categorical_crossentropy(label_h_b, logits)

        loss_history[i,idx] = loss_value.numpy().mean()

        grads = tape.gradient(loss_value, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
#Let's visualize the Loss function

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2,2)

fig.suptitle('How learning rate changes the loss function')

axs[0,0].plot(x,loss_history[0,:])

axs[0, 0].set_title('Learning rate: '+ str(learning_rate[0]))

axs[0,1].plot(x,loss_history[1,:])

axs[0, 1].set_title('Learning rate: '+ str(learning_rate[1]))

axs[1,0].plot(x,loss_history[2,:])

axs[1, 0].set_title('Learning rate: '+ str(learning_rate[2]))

axs[1,1].plot(x,loss_history[3,:])

axs[1, 1].set_title('Learning rate: '+ str(learning_rate[3]))

for ax in axs.flat:

    ax.set(xlabel='Epoch', ylabel='Loss')

for ax in axs.flat:

    ax.label_outer()