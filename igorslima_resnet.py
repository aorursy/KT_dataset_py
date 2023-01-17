import numpy as np
import pandas as pd

np.random.seed(2018)

%matplotlib inline
import matplotlib.pyplot as plt

import keras
from keras import regularizers
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten

import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
from keras.preprocessing.image import ImageDataGenerator
df_train = pd.read_csv('../input/dog-breed-identification/labels.csv') # file with the names of the images and its breed for training
df_test = pd.read_csv('../input/dog-breed-identification/sample_submission.csv') ## file with with the name of the images for testing
df_train.head()
df_test.head()
targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse = True)
one_hot_labels = np.asarray(one_hot)
im_size = 197
x_train = []
y_train = []
x_test = []
i = 0 
for f, breed in tqdm(df_train.values):
    img = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(f))
    x_train.append(cv2.resize(img, (im_size, im_size)))
    label = one_hot_labels[i]
    y_train.append(label)
    i += 1
del df_train
for f in tqdm(df_test['id'].values):
    img = cv2.imread('../input/dog-breed-identification/test/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (im_size, im_size)))
# Criando uma variável para saber a quantidade classes que temos no dataset
num_class = 120
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, shuffle=True,  test_size=0.2, random_state=1)
del x_train, y_train
datagen = ImageDataGenerator(width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.2,
                            rotation_range=30,
                            vertical_flip=False,
                            horizontal_flip=True)


datagen.fit(X_train)
def create_my_model(optimizer):
    base_model = ResNet50(weights="../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False, input_shape=(im_size, im_size, 3))
    dropout = base_model.output
    dropout = Dropout(0.5)(dropout)
    model_with_dropout = Model(inputs=base_model.input, outputs=dropout)
    
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(num_class, activation='softmax',
                        kernel_regularizer=regularizers.l2(0.0015),
                        activity_regularizer=regularizers.l1(0.0015))(x)
    
    
    
    my_model = Model(inputs=base_model.input, outputs=predictions)
    
#     for layer in my_model.layers:
#         layer.treinable = False
    
    my_model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
    return my_model

def gerar_grafico(historia, titulo):
    plt.plot(historia.history['acc'])
    plt.plot(historia.history['val_acc'])
    plt.title('Acurácia ' + titulo)
    plt.ylabel('Acurácia')
    plt.xlabel('Épocas')
    plt.legend(['treino', 'validação'], loc='upper left')
    plt.show()
    plt.plot(historia.history['loss'])
    plt.plot(historia.history['val_loss'])
    plt.title('Loss ' + titulo)
    plt.ylabel('Loss')
    plt.xlabel('Épocas')
    plt.legend(['treino', 'validação'], loc='upper left')
    plt.show()
train_generator = datagen.flow(np.array(X_train), np.array(Y_train), 
                               batch_size=32) 
model_rmsprop_com_regularizador = create_my_model(optimizer='rmsprop')
model_sgd_com_regularizador = create_my_model(optimizer='sgd')
history_rmsprop_com_regularizador = model_rmsprop_com_regularizador.fit_generator(
    train_generator,
    epochs=10, steps_per_epoch=len(X_train) / 18, #len(X_train) / 18,
    validation_data=(np.array(X_train), np.array(Y_train)), validation_steps=len(X_valid) / 18)

preds = model_rmsprop_com_regularizador.predict(np.array(x_test), verbose=1)

gerar_grafico(history_rmsprop_com_regularizador, 
              "ResNet50 with RMSprop")

sub = pd.DataFrame(preds)
col_names = one_hot.columns.values
sub.columns = col_names
sub.insert(0, 'id', df_test['id'])
sub.head(5)

sub.to_csv("predictions_resnet50_rmsprop.csv")

model_rmsprop_com_regularizador.save('resnet50_rmsprop.h5')
history_sgd_com_regularizador = model_sgd_com_regularizador.fit_generator(
    train_generator,
    epochs=10, steps_per_epoch=len(X_train) / 18, #len(X_train) / 18,
    validation_data=(np.array(X_train), np.array(Y_train)), validation_steps=len(X_valid) / 18)

preds = model_sgd_com_regularizador.predict(np.array(x_test), verbose=1)

gerar_grafico(history_sgd_com_regularizador, 
              "ResNet50 com SGD, data augmentation e Regularizador")

sub = pd.DataFrame(preds)
col_names = one_hot.columns.values
sub.columns = col_names
sub.insert(0, 'id', df_test['id'])
sub.head(5)

sub.to_csv("output_sgd_v2_com_data_augmentation_e_regularizador.csv")

model_sgd_com_regularizador.save('sgd_v2_com_data_augmentation_e_regularizador.h5')