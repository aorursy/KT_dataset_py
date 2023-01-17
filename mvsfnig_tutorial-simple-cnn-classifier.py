import numpy as np       # linear algebra
import pandas as pd      # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf  # lib AI and numerical computing with tensores

# keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, ZeroPadding2D
from keras.models import Sequential
from keras.optimizers import RMSprop

from matplotlib import pyplot as plt

from keras.utils.np_utils import to_categorical # data processing - convert to one-hot-encoding

import matplotlib.pyplot as plt # visualization
%matplotlib inline

import os # comunication with operational system
print(os.listdir("../input"))
# lendo o dataset 
data = pd.read_csv('../input/train.csv')

# labels
x = data.drop(labels=["label"], axis=1) 

# features
y = data['label']

# liberando mais espaco
del data
print(x.shape, y.shape)
# redimensiomado o array 784 em matrix de 28 x 28 em uma canal, imagem em tons de cinza
x = x.values.reshape(-1,28,28,1)
x.shape
fig,ax = plt.subplots(1,4, figsize=(12,5))

for i in range(4):
    ax[i].imshow(x[i][:,:,0], cmap="gray")
# as imagens são formadas de valores de 0 a 255, com essa rapida normalização, elas ficam na escala entre 0 e 1
x = x / 255.0

# é melhor para o modelo de aprendizado de máquina convergir valores de 0 a 1, do que 1 a 10, o aprendizado se torna mais rápido
y = to_categorical(y, num_classes=10)
print(x.shape, y.shape)
from sklearn.model_selection import train_test_split
# divide os dados de treino e validacao para setar no treinamento
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
del x
del y
def simpleCNN(entrada, weights_path=None):
    
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=entrada))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    if weights_path:
        model.load_weights(weights_path)

    return model

# definindo o model
model = simpleCNN(x_train[1].shape)

model.compile(RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('-'*30)
print('treinando o modelo')
log = model.fit(x_train, y_train, batch_size=10, epochs=3)

log.history.keys()
# summarize history for loss
plt.plot(log.history['acc'], '--go')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# summarize history for loss
plt.plot(log.history['loss'], '--ro')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Accuracy: %.2f ' %(test_acc*100))
print('Loss: %.2f '  %(test_loss*100))
# imagens utilizadas para teste

imagens_selecionadas = x_test[0:20,:,:,]

fig,ax = plt.subplots(1,imagens_selecionadas.shape[0], figsize=(12,5))

for i in range(imagens_selecionadas.shape[0]):
    ax[i].imshow(imagens_selecionadas[i,:,:,0], cmap="gray")
# fazendo a predição sobre os dados de teste
predicao = model.predict(imagens_selecionadas)
from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt =  'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
np.argmax(imagens_selecionadas)

y_pred = []
corretas = []

for i in range(len(imagens_selecionadas)):
    y_pred.append(np.argmax(predicao[i]))
    corretas.append(np.argmax(y_test[i]))
    
cnf_matrix = confusion_matrix(corretas, y_pred)
cnf_matrix
plot_confusion_matrix(cnf_matrix, classes=['1','2','3','4','5','6','7','8','9'])
# lendo o dataset 
data = pd.read_csv('../input/test.csv')
data.head()
# tranformando o vetor em imagem
test = data.values.reshape(-1,28,28,1)
test.shape
# normalizando os dados de teste
test = test /  255.0

# fazendo a prediação
previsoes_test = model.predict(test)
previsoes_test_label = []

for i in range(len(previsoes_test)):
    previsoes_test_label.append(np.argmax(previsoes_test[i]))
submissions=pd.DataFrame({"ImageId": list(range(1,len(previsoes_test_label)+1)),
                         "Label": previsoes_test_label})
submissions.to_csv("DR.csv", index=False, header=True)