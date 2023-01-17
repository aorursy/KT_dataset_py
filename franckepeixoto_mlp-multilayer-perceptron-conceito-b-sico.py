#carregar dataset mnist

import numpy as np

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_labels = len(np.unique(y_train))

print("total de labels:\t{}".format(num_labels))

print("labels:\t\t\t{0}".format(np.unique(y_train)))
#converter em one-hot

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
# Assumindo que nossa imagem é quadrada.

image_size = x_train.shape[1] 

input_size = image_size * image_size



print("x_train:\t{}".format(x_train.shape))

print("x_test:\t\t{}\n".format(x_test.shape))



print('Redimensionar e normalizar.\n')



x_train = np.reshape(x_train, [-1, input_size])

x_train = x_train.astype('float32') / 255

x_test = np.reshape(x_test, [-1, input_size])

x_test = x_test.astype('float32') / 255



print("x_train:\t{}".format(x_train.shape))

print("x_test:\t\t{}".format(x_test.shape))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Dropout



# Parâmetros

batch_size = 128 # É o tamanho da amostra de entradas a serem processadas em cada etapa de treinamento. //epocas

hidden_units = 256

dropout = 0.45



# Nossa  MLP com ReLU e Dropout 

model = Sequential()



model.add(Dense(hidden_units, input_dim=input_size))

model.add(Activation('relu'))

model.add(Dropout(dropout))



model.add(Dense(hidden_units))

model.add(Activation('relu'))

model.add(Dropout(dropout))



model.add(Dense(num_labels))
model.add(Activation('softmax'))



model.summary()
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='mlp.png', show_shapes=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)
print("MLP:\nValidar o modelo em nosso dataset de teste:\n")

_, acc = model.evaluate(x_test,

                        y_test,

                        batch_size=batch_size,

                        verbose=0)

print("\nAccuracy: %.1f%%\n" % (100.0 * acc))
print('Carregando novamente nosso dataset\n')

(x_train, _), (x_test, _) = mnist.load_data()



image_size = x_train.shape[1] 



print("x_train:\t{}".format(x_train.shape))

print("x_test:\t\t{}\n".format(x_test.shape))



print('Redimensionar e normalizar.\n')



x_train = np.reshape(x_train, [-1, image_size, image_size, 1])

x_train = x_train.astype('float32') / 255

x_test = np.reshape(x_test, [-1, image_size, image_size, 1])

x_test = x_test.astype('float32') / 255



print("x_train:\t{}".format(x_train.shape))

print("x_test:\t\t{}".format(x_test.shape))



input_shape = (image_size, image_size, 1)

print('\nProcessadas em escala de cinza.\n{}'.format(input_shape))

batch_size = 128



kernel_size = 3



filters = 64



dropout = 0.3
from tensorflow.keras.layers import Conv2D,Input,MaxPooling2D,Flatten

from tensorflow.keras.models import Model





inputs = Input(shape=input_shape)



y = Conv2D(filters=filters,

           kernel_size=kernel_size,

           activation='relu')(inputs)



y = MaxPooling2D()(y)#padrão pool_size=(2, 2)



y = Conv2D(filters=filters,

           kernel_size=kernel_size,

           activation='relu')(y)



y = MaxPooling2D()(y)



y = Conv2D(filters=filters,

           kernel_size=kernel_size,

           activation='relu')(y)



y = Flatten()(y)



y = Dropout(dropout)(y)



outputs = Dense(num_labels, activation='softmax')(y)



model = Model(inputs=inputs, outputs=outputs)



model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)
print("CNN:\nValidar o modelo em nosso dataset de teste:\n")

_, acc = model.evaluate(x_test,

                        y_test,

                        batch_size=batch_size,

                        verbose=0)

print("\nAccuracy: %.1f%%\n" % (100.0 * acc))