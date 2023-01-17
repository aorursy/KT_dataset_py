from glob import glob
import pandas as pd
#mete aqui a diretoria onde do teu exccel com os nomes das imagens e com as labels com 0 e 1 para má e boa qualidade
#o pd é para termos uma dataframe que é um objeto organizado em especie de excel com colunas e valores para cada coluna (faz print de labels para perceberes)
dataframe = pd.read_csv("../input/-pinho-1st-training/Treino1/database.csv", sep=";")
dataframe['status'] = dataframe['status'].apply(str)
print(dataframe.head())
print(dataframe.tail())

from sklearn.model_selection import train_test_split
#aqui fazes o split de dados de treino e teste (consoante o nome que deste às tuas colunas no dataframe metes dataframe.coluna_nome_imagens e dataframe.coluna_nome_labels)
#neste caso estao 10% para teste ou seja, para depois avaliares o modelo com dados desconhecidos
x_train, x_test, y_train, y_test = train_test_split(dataframe.image, dataframe.status, train_size=0.8, test_size=0.2)
d_train = {'image': x_train, 'status': y_train.astype(str)}
d_test = {'image': x_test, 'status': y_test.astype(str)}
df_train = pd.DataFrame(d_train)
df_test = pd.DataFrame(d_test)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
BATCH_SIZE = 16

#Estas variações é para trazer mais variabilidade aos dados de treino, 
#assim simula como se tivesses mais imagens do que tens apenas com pequenas variações
datagen_train = ImageDataGenerator(
    horizontal_flip=True,
    preprocessing_function=lambda x: x/127.5 - 1,
    brightness_range=[0.9, 1.1],
    vertical_flip=True,
    rotation_range=20,
)

#para o teste nao queres as colocar alterações, so mesmo o passo da normalização, porque queremos 
#ver como ele avalia as imagens como elas são
#para a preprocecssing_function vamos só normalizar os valores para estarem dentro de 0 e 1 
#em vez de 0 e 255 que é o normal
#lambda é apenas porque é uma função definida localmente
datagen_test = ImageDataGenerator(
    preprocessing_function=lambda x: x/127.5 - 1,
)

#o image e o label corresponde aos nomes das colunas que deste aos dataframes de treino e teste
#o batch_size é o numero de imagens que ele avalia de uma vez (em cada epoca o modelo percorre 
#todas as imagens por batches)
#Mete todas as imagens (boas e más) dentro de uma unica pasta porque queres ter os pinhoes 
#bons e maus no mesmo sitio

dic = "../input/-pinho-1st-training/Treino1/train"
G_train = datagen_train.flow_from_dataframe(
    dataframe=df_train,
    x_col="image",
    y_col="status",
    directory=dic,
    class_mode="binary",
    target_size=(241, 171),
    batch_size=BATCH_SIZE
)

G_test = datagen_test.flow_from_dataframe(
    dataframe=df_test,
    x_col="image",
    y_col="status",
    directory=dic,
    class_mode="binary",
    target_size=(241, 171),
    batch_size=BATCH_SIZE
)

import keras
LeakyReLU = keras.layers.LeakyReLU(alpha=0.01)
import cv2
import matplotlib.pyplot as plt
im = cv2.imread("../input/-pinho-1st-training/Treino1/train/500.jpg")
im.shape
plt.imshow(im)
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Dropout, Dense, Flatten
from tensorflow_addons.layers.normalizations import InstanceNormalization
from tensorflow.keras.models import Model

def d_layer(layer_input, filters, f_size=4, normalization=True, dropout_rate=0.2):
    """Discriminator layer"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    d = Dropout(dropout_rate)(d)
    if normalization:
        d = InstanceNormalization()(d)
    return d

img_shape = (241, 171, 3)

img = Input(shape=img_shape)

d1 = d_layer(img, 32, normalization=False)
d2 = d_layer(d1, 64)
d3 = d_layer(d2, 128)
d4 = d_layer(d3, 256)

#status_prediction = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
flatten = Flatten()(d4)
out = Dense(units=12, activation="relu")(flatten)
status_prediction = Dense(units=1, activation="softmax")(out)

model = Model(img, status_prediction)

model.summary()
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=32, kernel_size=(2, 2)))
model.add(LeakyReLU)
model.add(keras.layers.Dropout(rate=0.3))

model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3)))
model.add(LeakyReLU)
model.add(keras.layers.Dropout(rate=0.3))

model.add(keras.layers.Conv2D(filters=64, activation="relu", kernel_size=(3, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
model.add(keras.layers.Dropout(rate=0.3))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=12, activation="relu"))
model.add(keras.layers.Dense(units=1, activation="sigmoid"))
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

model.compile(optimizer=Adam, loss=binary_crossentropy, metrics=["binary_accuracy"])

model.fit_generator(G_train, epochs=20)
model.evaluate_generator(G_test)