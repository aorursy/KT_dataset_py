import pandas as pd

import matplotlib.pyplot as plt

from tensorflow import keras

from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import seaborn as sns



from random import choices
fashion_PATH = '/kaggle/input/fashionmnist/'



fashion_train = pd.read_csv(fashion_PATH+'fashion-mnist_train.csv')
X, y = fashion_train.iloc[:,1:].values/255, fashion_train.iloc[:,0].values
label = {

0:'T-shirt/top',

1:'Trouser',

2:'Pullover',

3:'Dress',

4:'Coat',

5:'Sandal',

6:'Shirt',

7:'Sneaker',

8:'Bag',

9:'Ankle boot'

}
N_images = 50



# localização dos exemplos na matriz de dados 

rows = choices(range(0, 60000), k=N_images)



# selecionando os dígitos, já no formato de matriz

digitos = [X[i].reshape(28,28) for i in rows]

label_value = y[rows]



# criando figura do matplotlib

fig, ax = plt.subplots(5, int(len(rows)/5),figsize=(18,10))



# plotando!

for i in range(len(rows)):

    j = int(i/10)

    k = i - j*10

    ax[j, k].imshow(digitos[i], cmap = plt.cm.binary, interpolation="nearest")

    ax[j, k].set_title(label[label_value[i]])

    ax[j, k].axis('off')
def compare_confusion_matriz(modelo, X_treino, X_validacao, y_treino, y_validacao):

    

    y_validacao_pred = modelo.predict(X_validacao)

    y_train_pred = modelo.predict(X_treino)

    

    if isinstance(m, keras.models.Sequential):

        y_validacao_pred = np.argmax(y_validacao_pred, axis=1)

        y_train_pred = np.argmax(y_train_pred, axis=1)

    

    confusao_val = confusion_matrix(y_validacao, y_validacao_pred)

    confusao_tr = confusion_matrix(y_treino, y_train_pred)

    

    fig, ax = plt.subplots(1, 2,figsize=(20,10))

    sns.heatmap(pd.DataFrame(confusao_val).rename(label, axis=1).rename(label, axis=0), ax=ax[0], cbar=False, annot=True)

    ax[0].set_title('Matriz de confusão validação', size=20)

    ax[0].set_yticklabels(ax[0].get_xticklabels(), rotation=0, size=15)

    ax[0].set_xticklabels(ax[0].get_yticklabels(), rotation=90, size=15)

    sns.heatmap(pd.DataFrame(confusao_tr), ax=ax[1], cbar=False, annot=True)

    ax[1].set_title('Matriz de confusão treino', size=20)

    ax[1].set_yticklabels(ax[1].get_xticklabels(), rotation=0, size=15)

    ax[1].set_xticklabels(ax[1].get_yticklabels(), rotation=0, size=15)

    plt.show()
def acuracia(modelo, X_treino, X_validacao, y_treino, y_validacao):



    y_validacao_pred = modelo.predict(X_validacao)

    y_train_pred = modelo.predict(X_treino)

    

    if isinstance(m, keras.models.Sequential):

        y_validacao_pred = np.argmax(y_validacao_pred, axis=1)

        y_train_pred = np.argmax(y_train_pred, axis=1)

    

    acc_tr = accuracy_score(y_treino, y_train_pred)

    acc_val = accuracy_score(y_validacao, y_validacao_pred)



    return {'Acurácia do treino': acc_tr, 'Acurácia da validação': acc_val}
def plot_erros(model, X, target):



    y_pred = model.predict(X)

    y_pred = np.argmax(y_pred, axis=1)

    

    predicao = pd.DataFrame(data={'predicao':y_pred, 'target':target})

    predicao_erros = predicao[predicao.predicao != predicao.target]

    

    N_images = 50



    # localização dos exemplos na matriz de dados 

    rows = predicao_erros.index[:N_images]



    # selecionando os dígitos, já no formato de matriz

    digitos = [X[i].reshape(28,28) for i in rows]

    label_value = predicao.loc[rows, 'target']

    label_errors = predicao.loc[rows, 'predicao']



    # criando figura do matplotlib

    fig, ax = plt.subplots(5, int(len(rows)/5),figsize=(30,10))



    # plotando!

    for i in range(len(rows)):

        j = int(i/10)

        k = i - j*10

        ax[j, k].imshow(digitos[i], cmap = plt.cm.binary, interpolation="nearest")

        ax[j, k].set_title(f'{label[label_value.iloc[i]]} confused {label[label_errors.iloc[i]]}')

        ax[j, k].axis('off')
X_treino, X_validacao, y_treino, y_validacao = train_test_split(X, y, test_size=0.2, random_state=0)
m = keras.models.Sequential()



m.add(keras.layers.Dense(200, input_shape = (784,), activation="relu"))

m.add(keras.layers.Dense(10, activation="softmax"))



m.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
%%time 



H = m.fit(X_treino, y_treino, 

          batch_size = 200, epochs = 10, 

          validation_data = (X_validacao, y_validacao));
compare_confusion_matriz(m, X_treino, X_validacao, y_treino, y_validacao)
acuracia(m, X_treino, X_validacao, y_treino, y_validacao)
plot_erros(m, X_validacao, y_validacao)
%%time 



neurons_list = [10, 20, 40, 100, 150, 200, 300, 400, 600, 1000]

acuracia_dict = dict()



for value in neurons_list:

    m = keras.models.Sequential()



    m.add(keras.layers.Dense(value, input_shape = (784,), activation="relu"))

    m.add(keras.layers.Dense(10, activation="softmax"))



    m.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    H = m.fit(X_treino, y_treino, 

              batch_size = 200, epochs = 20, 

              validation_data = (X_validacao, y_validacao));



    acuracia_dict[value] = acuracia(m, X_treino, X_validacao, y_treino, y_validacao)
pd.DataFrame(acuracia_dict).T.plot()

plt.xlabel('Número de neurônios na camada interna')

plt.ylabel('Acuracia')

plt.show()
%%time 



neurons_list = [10, 20, 40, 100, 200]

acuracia_dict = dict()



for value in neurons_list:

    m = keras.models.Sequential()



    m.add(keras.layers.Dense(400, input_shape = (784,), activation="relu"))

    m.add(keras.layers.Dense(value, input_shape = (400,), activation="relu"))

    m.add(keras.layers.Dense(10, activation="softmax"))



    m.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    H = m.fit(X_treino, y_treino, 

              batch_size = 200, epochs = 20, 

              validation_data = (X_validacao, y_validacao));



    acuracia_dict[value] = acuracia(m, X_treino, X_validacao, y_treino, y_validacao)
pd.DataFrame(acuracia_dict).T.plot()

plt.xlabel('Número de neurônios na camada interna')

plt.ylabel('Acuracia')

plt.show()
%%time 



batch_size_list = [50, 70, 100, 200, 300, 400]

acuracia_dict = dict()



for value in batch_size_list:

    m = keras.models.Sequential()



    m.add(keras.layers.Dense(400, input_shape = (784,), activation="relu"))

    m.add(keras.layers.Dense(10, activation="softmax"))



    m.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    H = m.fit(X_treino, y_treino, 

              batch_size = value, epochs = 20, 

              validation_data = (X_validacao, y_validacao));



    acuracia_dict[value] = acuracia(m, X_treino, X_validacao, y_treino, y_validacao)
pd.DataFrame(acuracia_dict).T.plot()

plt.xlabel('Batch Size')

plt.ylabel('Acuracia')

plt.show()
activation_list = ['relu', 'selu', 'tanh', 'sigmoid']

acuracia_dict = dict()



for activation in activation_list:

    m = keras.models.Sequential()



    m.add(keras.layers.Dense(400, input_shape = (784,), activation=activation))

    m.add(keras.layers.Dense(10, activation="softmax"))



    m.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    H = m.fit(X_treino, y_treino, 

              batch_size = 300, epochs = 20, 

              validation_data = (X_validacao, y_validacao));



    acuracia_dict[activation] = acuracia(m, X_treino, X_validacao, y_treino, y_validacao)
pd.DataFrame(acuracia_dict).T.plot.bar()

plt.xlabel('Activação')

plt.ylabel('Acuracia')

plt.ylim(0.80, 1)

plt.show()
optimizer_list = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']

acuracia_dict = dict()



for optimizer in optimizer_list:

    m = keras.models.Sequential()



    m.add(keras.layers.Dense(400, input_shape = (784,), activation='relu'))

    m.add(keras.layers.Dense(10, activation="softmax"))



    m.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



    H = m.fit(X_treino, y_treino, 

              batch_size = 300, epochs = 20, 

              validation_data = (X_validacao, y_validacao));



    acuracia_dict[optimizer] = acuracia(m, X_treino, X_validacao, y_treino, y_validacao)
pd.DataFrame(acuracia_dict).T.plot.bar()

plt.xlabel('Optimizer')

plt.ylabel('Acuracia')

plt.ylim(0.80, 1.05)

plt.show()
dropout_list = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9]

acuracia_dict = dict()



for dropout in dropout_list:

    m = keras.models.Sequential()



    m.add(keras.layers.Dense(400, input_shape = (784,), activation='relu'))

    m.add(keras.layers.Dropout(dropout))

    m.add(keras.layers.Dense(10, activation="softmax"))



    m.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])



    H = m.fit(X_treino, y_treino, 

              batch_size = 300, epochs = 20, 

              validation_data = (X_validacao, y_validacao));



    acuracia_dict[dropout] = acuracia(m, X_treino, X_validacao, y_treino, y_validacao)
pd.DataFrame(acuracia_dict).T.plot.bar()

plt.xlabel('Dropout')

plt.ylabel('Acuracia')

plt.ylim(0.79, 1.0)

plt.show()
epochs_list = [5, 7, 10, 12, 15, 20, 30, 40]

acuracia_dict = dict()



for epochs in epochs_list:

    m = keras.models.Sequential()



    m.add(keras.layers.Dense(400, input_shape = (784,), activation='relu'))

    m.add(keras.layers.Dropout(.5))

    m.add(keras.layers.Dense(10, activation="softmax"))



    m.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    H = m.fit(X_treino, y_treino, 

              batch_size = 300, epochs = epochs, 

              validation_data = (X_validacao, y_validacao));



    acuracia_dict[epochs] = acuracia(m, X_treino, X_validacao, y_treino, y_validacao)
pd.DataFrame(acuracia_dict).T.plot()

plt.xlabel('Epochs')

plt.ylabel('Acuracia')

plt.show()