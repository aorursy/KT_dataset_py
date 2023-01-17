import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
# Carregar Dataset
train=pd.read_csv('../input/digit-recognizer/train.csv')
test_images=pd.read_csv('../input/digit-recognizer/test.csv')
# Separar em "X_Train" e "Y_train"

# X_Train: axis=1 para dropar toda coluna target "label"
train_images=train.drop('label',axis=1)

#Y_Train
train_labels=train['label']
# Visualizar dataset
train_images.head()
# Missing Values ?
train_images.isnull().sum()
# Normalização dos dados para escala de cinza
train_images=train_images/255.0
test_images=test_images/255.0
# Modelamos os dados para 28x28x1, afinal, possuimos as imagens no formato de vetor 1D com 784 valores
train_images=train_images.values.reshape(len(train_images),28,28,1)
test_images=test_images.values.reshape(len(test_images),28,28,1)
# Podemos visualizar 1 amostra
plt.imshow(train_images[78][:,:,0])
# Carregar libs para modelo preditivo
import tensorflow as tf
from tensorflow import keras
def build_model(hp):
    model=keras.Sequential([
        keras.layers.Conv2D(
            filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),   # escolherá um valor do range
            kernel_size=hp.Choice('conv_1_kernel', values = [3,5,7]),                # escolherá apenas os melhores valores
            activation='relu',
            input_shape=(28,28,1)                                                    
        ),
         keras.layers.Conv2D(
            filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=16),
            kernel_size=hp.Choice('conv_2_kernel', values = [3,5,7]),
            activation='relu'
         ),
         keras.layers.Flatten(),
         keras.layers.Dense(
             units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
             activation='relu'
             ),
        keras.layers.Dense(10, activation='softmax')     
        ])
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
    return model
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
tuner_search = RandomSearch(build_model,max_trials=5,objective='val_accuracy')
# Precisão do número de tentativas e hiperparâmetros usados ​​para cada tentativa(pode levar muito tempo)
tuner_search.search(train_images,train_labels,epochs=3,validation_split=0.1,verbose=2)
# Escolhemos apenas os melhores modelos
model = tuner_search.get_best_models(num_models=1)[0]
model.summary()
# Treinamos
model.fit(train_images, train_labels, epochs=8,initial_epoch=3, validation_split=0.1,verbose=1)
# Preparamos predições

# Coluna ImageId
test_pred = pd.DataFrame(model.predict(test_images, batch_size=200))
test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))
test_pred.index.name = 'ImageId'

# Coluna "Label"
test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()
test_pred['ImageId'] = test_pred['ImageId'] + 1

test_pred.head()
test_pred.to_csv('submission.csv', index = False)
