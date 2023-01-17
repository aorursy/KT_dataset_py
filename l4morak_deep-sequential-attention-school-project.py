import keras

from keras.datasets import fashion_mnist, mnist, cifar10

from keras.models import *

from keras.layers import *

from keras.optimizers import RMSprop

from keras.constraints import *

from keras.utils import plot_model

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



batch_size = 128

num_classes = 10

epochs = 20



# the data, split between train and test sets

(x_train, y_train), (x_test, y_test) = cifar10.load_data()



x_train = x_train.reshape(-1, 32, 32, 3)

x_test = x_test.reshape(10000, 32, 32, 3)

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)



# Any results you write to the current directory are saved as output.
inp = Input((32, 32, 3 ))

x = inp

for a in range(2):

    x = Conv2D(8, (3,3))(x)

    x = Conv2D(8, (3,3))(x)

    x = Conv2D(8, (3,3))(x)

    x = MaxPooling2D()(x)

x = Flatten()(x)

x = Dense(200)(x)

x = Dense(10)(x)

x = Activation('softmax')(x)

model_for_weights = Model(inp, x)



model_for_weights.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model_for_weights.fit(x_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,  

                    verbose=1,

                    validation_data=(x_test, y_test))





inp = Input((32, 32, 3 ))

x = inp

for a in range(2):

    x = Conv2D(8, (3,3))(x)

    x = Conv2D(8, (3,3))(x)

    x = Conv2D(8, (3,3))(x)

    x = MaxPooling2D()(x)

x = Flatten()(x)

x = Dense(200)(x) 

x = Dense(200)(x)

xa = Dense(200, activation = 'softmax')(x)

x = multiply([x, xa])

x = Dense(10)(x)

x = Activation('softmax')(x)

model_sa = Model(inp, x)



for a in range(1,4):

    model_sa.layers[a].set_weights(model_for_weights.layers[a].get_weights())

for a in range(5,8):    

    model_sa.layers[a].set_weights(model_for_weights.layers[a].get_weights())

    

model_sa.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history_SA = model_sa.fit(x_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,  

                    verbose=1,

                    validation_data=(x_test, y_test))

score = model_sa.evaluate(x_test, y_test, verbose=0)



attention_depth = 4  

attention_layer = Dense(50, activation = 'softmax')

attention_memory_layer = Dense(50)



# build model

inp = Input((32, 32, 3 ))

x = inp

for a in range(2):

    x = Conv2D(8, (3,3))(x)

    x = Conv2D(8, (3,3))(x)

    x = Conv2D(8, (3,3))(x)

    x = MaxPooling2D()(x)

x = Flatten()(x)

x1 = Dense(50)(x)

attention = attention_layer(x1)

never_recount = RepeatVector(attention_depth)(x1)

attentions = Lambda(lambda y: y[:, 0, :])(never_recount)

memory = attention_memory_layer(attention)

memory = BatchNormalization(name = 'BN_0')(memory)

for a in range(1, attention_depth):

    extracted_mem = Lambda(lambda x: x[:, a, :] - Activation('sigmoid')(memory), name = 'Memory_extraction_n_' + str(a))(never_recount)

    attention = attention_layer(extracted_mem)

    attentions = concatenate([attentions, attention], name = 'concatenate_attentions_{0}_{1}'.format(a-1, a))

    memory = add([memory, attention_memory_layer(attention)])

    memory = BatchNormalization(name = 'BN_{0}'.format(a))(memory)

    

attentions = Reshape((attention_depth, 50), name = 'Reshape_to_sequence')(attentions)

attented = multiply([x1, attentions], name = 'Multiplication')

x = Bidirectional(SimpleRNN(50, name = 'Bi-RNN'))(attented) # Can be also LSTM or GRU, but I want to keep 20k params.

x = Dense(10)(x)

x = Activation('softmax')(x)

model_dsa = Model(inp, x)



for a in range(1,4):

    model_dsa.layers[a].set_weights(model_for_weights.layers[a].get_weights())

for a in range(5,8):    

    model_dsa.layers[a].set_weights(model_for_weights.layers[a].get_weights())



model_dsa.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')



history_DSA = model_dsa.fit(x_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,

                    validation_data=(x_test, y_test))
import matplotlib.pyplot as plt

plt.cla()

plt.plot(history_SA.history['val_acc'], color = 'r', label = 'Baseline with SA')

plt.plot(history_DSA.history['val_acc'], color = 'b', label = 'DSA')

plt.title('CIFAR with Conv')

plt.xlabel('Epoch')

plt.ylabel('Validation accuracy')

plt.legend()

plt.show()



plt.cla()

plt.plot(history_SA.history['val_loss'], color = 'r', label = 'Baseline with SA')

plt.plot(history_DSA.history['val_loss'], color = 'b', label = 'DSA')

plt.title('CIFAR with Conv')

plt.xlabel('Epoch')

plt.ylabel('Validation accuracy')

plt.legend()

plt.show()
from IPython.display import Image

plot_model(model_dsa, show_layer_names=True, to_file='model_plot.png', dpi = 52)

Image(filename='model_plot.png')
model_sa.summary()
model_dsa.summary()