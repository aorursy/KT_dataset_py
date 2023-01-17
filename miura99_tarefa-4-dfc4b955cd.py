#Carregando as imagens:
import numpy as np
ds = {
      "Pure":    np.load("../input/atividade-4-versao-2-fashion-mnist/train_images_pure.npy"),
      "Noisy":   np.load("../input/atividade-4-versao-2-fashion-mnist/train_images_noisy.npy"),
      "Rotated": np.load("../input/atividade-4-versao-2-fashion-mnist/train_images_rotated.npy"),
      "Both":    np.load("../input/atividade-4-versao-2-fashion-mnist/train_images_both.npy")
     }
import pandas as pd
#Carregando dataset de treino
train_info = pd.read_csv("../input/atividade-4-versao-2-fashion-mnist/train_labels.csv")
y_total = np.array(train_info['label'])
train_info['label'].value_counts()
train_info.info()
%matplotlib inline
import matplotlib.pyplot as plt
#Função para visualizar as 4 primeiras imagens de cada dataset de forma organizada
def head(ds, key):
    plt.subplot(341)
    plt.title(key)
    for i in range (4):
        plt.subplot(341 + i)
        plt.imshow(ds[i], cmap=plt.get_cmap('gray'))
    plt.show()
for key,item in ds.items():
    head(ds[key], key)
for key, item in ds.items():
    print(key, ds[key].shape)
#Importando bibliotecas que serão utilizadas do framework Keras:
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, LSTM, MaxPool2D, UpSampling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model
K.set_image_dim_ordering('th')
#Settando seed para o numpy:
seed = 42
np.random.seed(seed)
# input layer
input_layer = Input(shape=(1,28,28))

# encoding architecture
encoded_layer1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
encoded_layer1 = MaxPool2D( (2, 2), padding='same')(encoded_layer1)
encoded_layer2 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded_layer1)
encoded_layer2 = MaxPool2D( (2, 2), padding='same')(encoded_layer2)
encoded_layer3 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded_layer2)
latent_view    = MaxPool2D( (2, 2), padding='same')(encoded_layer3)

# decoding architecture
decoded_layer1 = Conv2D(16, (3, 3), activation='relu', padding='same')(latent_view)
decoded_layer1 = UpSampling2D((2, 2))(decoded_layer1)
decoded_layer2 = Conv2D(32, (3, 3), activation='relu', padding='same')(decoded_layer1)
decoded_layer2 = UpSampling2D((2, 2))(decoded_layer2)
decoded_layer3 = Conv2D(64, (3, 3), activation='relu')(decoded_layer2)
decoded_layer3 = UpSampling2D((2, 2))(decoded_layer3)
output_layer   = Conv2D(1, (3, 3), padding='same')(decoded_layer3)

# model compiling
autenc = Model(input_layer, output_layer)
autenc.compile(optimizer='adam', loss='mse')
autenc.summary()
from sklearn.model_selection import train_test_split
#Separando datasets para treino e validação. Aqui, estamos tratando o dataset puro como "variável reposta".
Xe_train, Xe_val, Ye_train, Ye_val = train_test_split(ds["Both"], ds["Pure"], test_size=0.2, random_state = seed)
Xe_train = Xe_train.reshape(Xe_train.shape[0], 1, 28, 28).astype('float32')
Ye_train = Ye_train.reshape(Ye_train.shape[0], 1, 28, 28).astype('float32')
Xe_val = Xe_val.reshape(Xe_val.shape[0], 1, 28, 28).astype('float32')
Ye_val = Ye_val.reshape(Ye_val.shape[0], 1, 28, 28).astype('float32')
Xe_train /= 255
Ye_train /= 255
Xe_val /= 255
Ye_val /= 255
#Usando early-stopping:
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=5, mode='auto')
history = autenc.fit(Xe_train, Ye_train, epochs=100, batch_size=1000, validation_data=(Xe_val, Ye_val), callbacks=[early_stopping])
#Visualizando o que o autoencoder faz:
foo = [Xe_val[i].reshape(28,28) for i in range(5)]
head(foo, "Ruidoso")
pred = autenc.predict(Xe_val[:5])
pred = [pred[i].reshape(28, 28) for i in range(pred.shape[0])]
head(pred, "Predição")
real = [Ye_val[i].reshape(28, 28) for i in range(5)]
head(real, "Real")
from sklearn.model_selection import train_test_split
#Base de treino a ser usada: output do autoencoder:
X_total = ds['Both']
X_total = X_total.reshape(X_total.shape[0], 1, 28, 28).astype('float32')
X_total /= 255
X_pred = autenc.predict(X_total)
X_pred = X_total
#Criando base de treino e validação (a ser usada com os datasets corrompidos)
X_train, X_val, y_train, y_val = train_test_split(X_pred, y_total, test_size=0.2, random_state = seed)
#Usando one-hot encoding para a saída:
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
num_classes = y_train.shape[1]
#Definindo o modelo (seguindo o tutorial de Keras, a otimização dos hiperparâmetros será feita na sequência):
def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(1, 28, 28), activation='tanh'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#Construindo o modelo
model = baseline_model()
model.summary()
#Treinando o modelo em cima do output do autoencoder. Válido ressaltar que foi feito train-validation split,
#para evitar overfitting.
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=1000, verbose=1, callbacks = [early_stopping])
#Testando modelo na base com imagens rotacionadas (usando auto encoder)
__ , X_test_rot = train_test_split(ds["Rotated"], test_size=0.2, random_state = seed)
X_test_rot = X_test_rot.reshape(X_test_rot.shape[0], 1, 28, 28).astype('float32')
X_test_rot /= 255
X_test_rot = autenc.predict(X_test_rot)
scores = model.evaluate(X_test_rot, y_val, verbose=0)
print("CNN Error for rotated test dataset: %.2f%%" % (100-scores[1]*100))
#Testando modelo na base com imagens rotacionadas e corrompidas 
__ , X_test_both = train_test_split(ds["Both"], test_size=0.2, random_state = seed)
X_test_both = X_test_both.reshape(X_test_both.shape[0], 1, 28, 28).astype('float32')
X_test_both /= 255
X_test_both = autenc.predict(X_test_both)
scores = model.evaluate(X_test_both, y_val, verbose=0)
print("CNN Error for rotated and noisy test dataset: %.2f%%" % (100-scores[1]*100))
__ , X_test_pure = train_test_split(ds["Pure"], test_size=0.2, random_state = seed)
X_test_pure = X_test_pure.reshape(X_test_pure.shape[0], 1, 28, 28).astype('float32')
X_test_pure /= 255
X_test_pure = autenc.predict(X_test_pure)
scores2 = model.evaluate(X_test_pure, y_val, verbose = 0)
print("CNN Error for pure test dataset: %.2f%%" % (100-scores2[1]*100))
def opt_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(1, 28, 28), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), input_shape=(1, 28, 28), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
optm = opt_model()
optm.summary()
optm.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=1000, verbose=1, callbacks = [early_stopping])
__ , X_test_rot = train_test_split(ds["Rotated"], test_size=0.2, random_state = seed)
X_test_rot = X_test_rot.reshape(X_test_rot.shape[0], 1, 28, 28).astype('float32')
X_test_rot /= 255
X_test_rot = autenc.predict(X_test_rot)
scores = optm.evaluate(X_test_rot, y_val, verbose=0)
print("DCNN Error for rotated test dataset: %.2f%%" % (100-scores[1]*100))
#Testando modelo na base com imagens rotacionadas e corrompidas 
__ , X_test_both = train_test_split(ds["Both"], test_size=0.2, random_state = seed)
X_test_both = X_test_both.reshape(X_test_both.shape[0], 1, 28, 28).astype('float32')
X_test_both /= 255
X_test_both = autenc.predict(X_test_both)
scores = optm.evaluate(X_test_both, y_val, verbose=0)
print("DCNN Error for rotated and noisy test dataset: %.2f%%" % (100-scores[1]*100))
#Carregando base de teste
X_test = np.load("../input/atividade-4-versao-2-fashion-mnist/Test_images.npy")
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
X_test = X_test / 255
#Transformando base de teste:
X_test = autenc.predict(X_test)
y_pred = optm.predict_classes(X_test)
output_data = {"Id": np.arange(0,10000), 
               "label": y_pred
              }

output = pd.DataFrame(data = output_data)
output.to_csv("output.csv", index = False)
