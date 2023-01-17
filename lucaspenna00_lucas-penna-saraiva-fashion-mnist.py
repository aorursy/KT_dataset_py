%matplotlib inline

import numpy as np#arrays operations
import matplotlib.pyplot as plt # shows the image
img_np1 = np.load("../input/train_images_pure.npy", mmap_mode=None, allow_pickle=True, fix_imports=True)
fig=plt.figure(figsize=(20, 20))
columns = 10
rows = 10
image_index = 0
for i in range(1, columns*rows +1):
    img = img_np1[image_index]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap="Greys")
    image_index = image_index + 1
    
plt.show()
img_np2 = np.load("../input/train_images_rotated.npy", mmap_mode=None, allow_pickle=True, fix_imports=True)

fig=plt.figure(figsize=(20, 20))
columns = 10
rows = 10
image_index = 0
for i in range(1, columns*rows +1):
    img = img_np2[image_index]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap="Greys")
    image_index = image_index + 1
    
plt.show()
img_np3 = np.load("../input/train_images_noisy.npy", mmap_mode=None, allow_pickle=True, fix_imports=True)

fig=plt.figure(figsize=(20, 20))
columns = 10
rows = 10
image_index = 0
for i in range(1, columns*rows +1):
    img = img_np3[image_index]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap="Greys")
    image_index = image_index + 1
    
plt.show()
img_np4 = np.load("../input/train_images_both.npy", mmap_mode=None, allow_pickle=True, fix_imports=True)

fig=plt.figure(figsize=(20, 20))
columns = 10
rows = 10
image_index = 0
for i in range(1, columns*rows +1):
    img = img_np4[image_index]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap="Greys")
    image_index = image_index + 1
    
plt.show()
test = np.load("../input/Test_images.npy", mmap_mode=None, allow_pickle=True, fix_imports=True)

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import pandas as pd
K.set_image_dim_ordering('th')

# Carregando dados de treino da base rotacionada

x_train_rotated = np.load("../input/train_images_rotated.npy", mmap_mode=None, allow_pickle=True, fix_imports=True)

# Carregando dados de treino da base both

x_train_both = np.load("../input/train_images_both.npy", mmap_mode=None, allow_pickle=True, fix_imports=True)

# Carregando dados dos labels

y_train = pd.read_csv('../input/train_labels.csv')
y_train = np.array(y_train.drop(['Id'], axis=1))

# Carreando dados de teste

x_test = np.load("../input/Test_images.npy", mmap_mode=None, allow_pickle=True, fix_imports=True)

# imagens podem ter várias camadas de cor - R,G,B, por exemplo. Assim sendo, modelos convolucionais geralmente esperam
# em suas entradas, matrizes tetradimensionais, i.e., no formato [samples][canais][width][height] , onde canais é o número
# de canais de cor de sua imagem. No caso do MNIST, é uma imagem em grayscale, então temos apenas um canal de cor. 

x_train_rotated = x_train_rotated.reshape(x_train_rotated.shape[0], 1, 28, 28).astype('float32')
x_train_both = x_train_both.reshape(x_train_both.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

## NORMALIZAR OS DADOS PARA MELHORAR O DESEMPENHO DA CONVOLUÇÃO
x_train_rotated = x_train_rotated / 255
x_train_both = x_train_both / 255
x_test = x_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
num_classes = y_train.shape[1]
print("Numero de classes: %d " % num_classes)

# separando o treinamento entre treino e validação: 
x_real_train_rotated, x_validation_rotated, y_real_train_rotated, y_validation_rotated = train_test_split(x_train_rotated, y_train, test_size = 0.2)
x_real_train_both, x_validation_both, y_real_train_both, y_validation_both = train_test_split(x_train_both, y_train, test_size = 0.2)
rnc1_model = Sequential()

rnc1_model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))

rnc1_model.add(Conv2D(15, (5, 5), activation='relu'))

rnc1_model.add(Dropout(0.2))

rnc1_model.add(Flatten())

rnc1_model.add(Dense(128, activation='relu'))

rnc1_model.add(Dense(50, activation='relu'))

rnc1_model.add(Dense(num_classes, activation='softmax'))

# Compile model
rnc1_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
rnc1_model.summary()

# Visualizing the model
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plot_model(rnc1_model, show_shapes=True, show_layer_names=True, to_file="cnn1.png")
img = mpimg.imread('cnn1.png')
plt.imshow(img)
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]

# Fit the model do treino ROTACIONADO
rnc1_model.fit(x_real_train_rotated, y_real_train_rotated, validation_data=(x_validation_rotated,y_validation_rotated), epochs=2, 
          batch_size=200, verbose=1, callbacks = callbacks)
# Fit the model do treino BOTH

rnc1_model.fit(x_real_train_both, y_real_train_both, validation_data=(x_validation_both,y_validation_both), epochs=2, 
          batch_size=200, verbose=1, callbacks = callbacks)
rnc2_model = Sequential()

rnc2_model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
rnc2_model.add(MaxPooling2D(pool_size=(2, 2)))

rnc2_model.add(Conv2D(15, (5, 5), activation='relu'))
rnc2_model.add(MaxPooling2D(pool_size=(2, 2)))

rnc2_model.add(Dropout(0.2))

rnc2_model.add(Flatten())

rnc2_model.add(Dense(128, activation='relu'))

rnc2_model.add(Dense(50, activation='relu'))

rnc2_model.add(Dense(num_classes, activation='softmax'))

# Compile model
rnc2_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
rnc2_model.summary()

# Visualizing the model
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plot_model(rnc2_model, show_shapes=True, show_layer_names=True, to_file="cnn2.png")
img = mpimg.imread('cnn2.png')
plt.imshow(img)
# Fit the model do treino ROTACIONADO
rnc2_model.fit(x_real_train_rotated, y_real_train_rotated, validation_data=(x_validation_rotated,y_validation_rotated), epochs=2, 
          batch_size=200, verbose=1, callbacks = callbacks)
# Fit the model do treino BOTH

rnc2_model.fit(x_real_train_both, y_real_train_both, validation_data=(x_validation_both,y_validation_both), epochs=2, 
          batch_size=200, verbose=1, callbacks = callbacks)
import numpy as np
import matplotlib.pyplot as plt

## carregar dados da base de treino noisy

x_train_noisy = np.load("../input/train_images_noisy.npy", mmap_mode=None, allow_pickle=True, fix_imports=True)
import cv2 #Importar OpenCV para processar imagem

# Remove noise
# Gaussian

no_noise = []

for i in range(len(x_train_noisy)):
    blur = cv2.GaussianBlur(x_train_noisy[i], (5, 5), 0)
    no_noise.append(blur)

x_train_no_noisy = no_noise

x_train_no_noisy = np.array(x_train_no_noisy)
## Mostrar a base de dados borradas "Gaussianamente"

fig=plt.figure(figsize=(20, 20))
columns = 10
rows = 10
image_index = 0
for i in range(1, columns*rows +1):
    img = x_train_no_noisy[image_index]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap="Greys")
    image_index = image_index + 1
    
plt.show()
# Reformando os dados

x_train_no_noisy = x_train_no_noisy.reshape(x_train_no_noisy.shape[0], 1, 28, 28).astype('float32')

x_train_noisy = x_train_noisy.reshape(x_train_noisy.shape[0], 1, 28, 28).astype('float32')

# Normalizando os dados

x_train_no_noisy = x_train_no_noisy / 255

x_train_noisy = x_train_noisy / 255

# Separando o treinamento entre treino e validação: 

x_real_train_no_noisy, x_validation_no_noisy, y_real_train_no_noisy, y_validation_no_noisy = train_test_split(x_train_no_noisy, y_train, test_size = 0.2)

x_real_train_noisy, x_validation_noisy, y_real_train_noisy, y_validation_noisy = train_test_split(x_train_noisy, y_train, test_size = 0.2)

# Fit the model do treino BORRADO
rnc1_model.fit(x_real_train_no_noisy, y_real_train_no_noisy, validation_data=(x_validation_no_noisy, y_validation_no_noisy), epochs=2, 
          batch_size=200, verbose=1, callbacks = callbacks)
# Fit the model do treino RUIDOSO
rnc1_model.fit(x_real_train_noisy, y_real_train_noisy, validation_data=(x_validation_noisy,y_validation_noisy), epochs=2, 
          batch_size=200, verbose=1, callbacks = callbacks)