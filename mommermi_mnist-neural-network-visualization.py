import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.animation

from sklearn.model_selection import train_test_split

plt.rcParams["animation.html"] = "jshtml"



from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.utils import to_categorical, plot_model

from keras.datasets import mnist

from keras import backend as K
# load mnist dataset

data = pd.read_csv('../input/train.csv')



# split data into train and test sample

x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:785], data.iloc[:,0],  

                                                    test_size = 0.1, random_state = 42)



x_train = x_train.values.reshape(37800, 784)

x_test = x_test.values.reshape(4200, 784)



# compute the number of labels

num_labels = len(np.unique(y_train))



# convert to one-hot vector

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)



# normalize

x_train = x_train.astype('float32') / 255

x_test = x_test.astype('float32') / 255



# network parameters

input_size = x_train.shape[1]

batch_size = 64

dropout = 0.45
# this model is a 3-layer MLP with ReLU and dropout after each layer

model = Sequential()

model.add(Dense(256, input_dim=input_size))

model.add(Activation('relu'))

model.add(Dropout(dropout))

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(dropout))

model.add(Dense(num_labels))

model.add(Activation('softmax'))

model.summary()



# loss function for one-hot vector using adam optimizer

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
# train the network

model.fit(x_train, y_train, epochs=20, batch_size=batch_size)



# validate the model on test dataset to determine generalization

loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print("\nTest accuracy: %.1f%%" % (100.0 * acc))
get_layer_output = K.function([model.layers[0].input, model.layers[0].input, model.layers[0].input],

                              [model.layers[1].output, model.layers[4].output, model.layers[7].output])



layer1_output, layer2_output, layer3_output = get_layer_output([x_train])
train_ids = [np.arange(len(y_train))[y_train[:,i] == 1] for i in range(10)]
%%capture

%matplotlib inline



# digit to be plotted

digit = 5



# indices of frames to be plotted for this digit

n = range(50)



# initialize plots

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15,4))



# prepare plots

ax1.set_title('Input Layer', fontsize=16)

ax1.axes.get_xaxis().set_visible(False)

ax1.axes.get_yaxis().set_visible(False)



ax2.set_title('Hidden Layer 1', fontsize=16)

ax2.axes.get_xaxis().set_visible(False)

ax2.axes.get_yaxis().set_visible(False)



ax3.set_title('Hidden Layer 2', fontsize=16)

ax3.axes.get_xaxis().set_visible(False)

ax3.axes.get_yaxis().set_visible(False)

    

ax4.set_title('Output Layer', fontsize=16)

ax4.axes.get_xaxis().set_visible(False)

ax4.axes.get_yaxis().set_visible(False)   



# add numbers to the output layer plot to indicate label

for i in range(3):

    for j in range(4):

        text = ax4.text(j, i, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, '', '']][i][j],

                        ha="center", va="center", color="w", fontsize=16)    

        

def animate(id):

    # plot elements that are changed in the animation

    digit_plot = ax1.imshow(x_train[train_ids[digit][id]].reshape((28,28)), animated=True)

    layer1_plot = ax2.imshow(layer1_output[train_ids[digit][id]].reshape((16,16)), animated=True)

    layer2_plot = ax3.imshow(layer2_output[train_ids[digit][id]].reshape((8,8)), animated=True)

    output_plot = ax4.imshow(np.append(layer3_output[train_ids[digit][id]], 

                                       [np.nan, np.nan]).reshape((3,4)), animated=True)

    return digit_plot, layer1_plot, layer2_plot, output_plot,



# define animation

ani = matplotlib.animation.FuncAnimation(f, animate, frames=n, interval=100)
ani
%%capture

%matplotlib inline



# digit to be plotted

digit = 6



# numbers of frames to be summed over

n = np.append([1], np.linspace(5, 100, 20, dtype=int))



# initialize plots

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15,4))



# add a counter indicating the number of frames used in the summation

counter = ax1.text(1, 2, 'n={}'.format(0), color='white', fontsize=16, animated=True)



# prepare plots

ax1.set_title('Input Layer', fontsize=16)

ax1.axes.get_xaxis().set_visible(False)

ax1.axes.get_yaxis().set_visible(False)



ax2.set_title('Hidden Layer 1', fontsize=16)

ax2.axes.get_xaxis().set_visible(False)

ax2.axes.get_yaxis().set_visible(False)



ax3.set_title('Hidden Layer 2', fontsize=16)

ax3.axes.get_xaxis().set_visible(False)

ax3.axes.get_yaxis().set_visible(False)

    

ax4.set_title('Output Layer', fontsize=16)

ax4.axes.get_xaxis().set_visible(False)

ax4.axes.get_yaxis().set_visible(False)   



# add numbers to the output layer plot to indicate label

for i in range(3):

    for j in range(4):

        text = ax4.text(j, i, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, '', '']][i][j],

                        ha="center", va="center", color="w", fontsize=16)    

        

def animate(id):

    # plot elements that are changed in the animation

    digit_plot = ax1.imshow(np.sum(x_train[train_ids[digit][:id]], axis=0).reshape((28,28)), animated=True)

    layer1_plot = ax2.imshow(np.sum(layer1_output[train_ids[digit][:id]], axis=0).reshape((16,16)), animated=True)

    layer2_plot = ax3.imshow(np.sum(layer2_output[train_ids[digit][:id]], axis=0).reshape((8,8)), animated=True)

    output_plot = ax4.imshow(np.append(np.sum(layer3_output[train_ids[digit][:id]], axis=0), 

                                       [np.nan, np.nan]).reshape((3,4)), animated=True)

    counter.set_text('n={}'.format(id))

    return digit_plot, layer1_plot, layer2_plot, output_plot, counter,



# define animation

ani = matplotlib.animation.FuncAnimation(f, animate, frames=n, interval=100)
ani
f, ax_arr = plt.subplots(2, 5, figsize=(15,10))



f.subplots_adjust(wspace=0.05, bottom=0.5, top=0.95)



for i, ax in enumerate(np.ravel(ax_arr)):

    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)

    if i <= 10:

        ax.set_title('- {} -'.format(i), fontsize=16)

        layer1_plot = ax.imshow(np.sum(layer1_output[train_ids[i]], axis=0).reshape((16,16)))
similarity_layer1 = np.zeros((10,10))



for i in range(10):

    for j in range(10):

        sum_i_normalized = np.sqrt(np.sum(layer1_output[train_ids[i]], axis=0)/np.sum(layer1_output[train_ids[i]]))

        sum_j_normalized = np.sqrt(np.sum(layer1_output[train_ids[j]], axis=0)/np.sum(layer1_output[train_ids[j]]))

        similarity_layer1[i,j] = np.sum(sum_i_normalized*sum_j_normalized)
f, ax = plt.subplots()



similarity_layer1_plot = ax.imshow(similarity_layer1, origin='lower')

plt.colorbar(similarity_layer1_plot)
f, ax_arr = plt.subplots(2, 5, figsize=(15,10))



f.subplots_adjust(wspace=0.05, bottom=0.5, top=0.95)



for i, ax in enumerate(np.ravel(ax_arr)):

    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)

    if i <= 10:

        ax.set_title('- {} -'.format(i), fontsize=16)

        layer2_plot = ax.imshow(np.sum(layer2_output[train_ids[i]], axis=0).reshape((8,8)))
similarity_layer2 = np.zeros((10,10))



for i in range(10):

    for j in range(10):

        sum_i_normalized = np.sqrt(np.sum(layer2_output[train_ids[i]], axis=0)/np.sum(layer2_output[train_ids[i]]))

        sum_j_normalized = np.sqrt(np.sum(layer2_output[train_ids[j]], axis=0)/np.sum(layer2_output[train_ids[j]]))

        similarity_layer2[i,j] = np.sum(sum_i_normalized*sum_j_normalized)
f, ax = plt.subplots()



similarity_layer2_plot = ax.imshow(similarity_layer2, origin='lower')

plt.colorbar(similarity_layer2_plot)