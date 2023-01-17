# comandos mágicos que não se comunicam com a linguagem Python e sim diretamente com o kernel do Jupyter
# começam com %

%load_ext autoreload
%autoreload 2

%matplotlib inline
# importando os principais módulos que usaremos ao longo da aula

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

import sklearn.neural_network
import sklearn.model_selection
import sklearn.metrics

from tensorflow import keras
mnist_PATH = '/kaggle/input/digit-recognizer/'

mnist_train = pd.read_csv(mnist_PATH+'train.csv')
mnist_test = pd.read_csv(mnist_PATH+'test.csv')
mnist_train
X, y = mnist_train.iloc[:,1:].values/255, mnist_train.iloc[:,0].values

X.shape, y.shape
# localização dos exemplos na matriz de dados 
loc = [0,1,20,34,54,659,541,5200,11200,16721,23000,24010,29050,30000,32000,34990,36000,37000,39000,41000]

# selecionando os dígitos, já no formato de matriz
digitos = [X[i].reshape(28,28) for i in loc]

# criando figura do matplotlib
fig, ax = plt.subplots(1,len(loc),figsize=(18,0.5))

# plotando!
[ax[i].imshow(digitos[i], cmap = matplotlib.cm.binary, interpolation="nearest") for i in range(len(loc))]

# desligando os eixos de todos os dígitos
[ax[i].axis('off') for i in range(len(loc))];
X_treino, X_validacao, y_treino, y_validacao = sklearn.model_selection.train_test_split(X, y, 
                                                                                        test_size=0.1, 
                                                                                        random_state=0)

y_treino.shape, y_validacao.shape
fig, ax = plt.subplots(1,2,figsize=(13,4))

sns.countplot(y_treino,ax=ax[0])
sns.countplot(y_validacao,ax=ax[1])

ax[0].set_title('Treino')
ax[1].set_title('Validação')

fig.suptitle('Proporção de classes (dígitos)');
m = sklearn.neural_network.MLPClassifier(random_state=0)

%time m.fit(X_treino, y_treino)
y_validacao_pred = m.predict(X_validacao)
print(y_validacao)
print(y_validacao_pred)
acc_tr = sklearn.metrics.accuracy_score(y_treino, m.predict(X_treino))
acc_val = sklearn.metrics.accuracy_score(y_validacao, y_validacao_pred)

print(f'Acurácia do treino: {acc_tr}')
print(f'Acurácia da validação: {acc_val}')
confusao = sklearn.metrics.confusion_matrix(y_validacao, y_validacao_pred)
confusao
def display_score(m):
        
    X = [X_treino, X_validacao]
    y = [y_treino, y_validacao]

    labels = ['Treino', 'Validação']
    
    if isinstance(m, keras.models.Sequential):
        if any([isinstance(l,keras.layers.Conv2D) for l in m.layers]):
            X = [X[i].reshape(-1,28,28,1) for i in (0,1)]
    
    y_pred = [m.predict(X[i]) for i in (0,1)]
    
    if isinstance(m, keras.models.Sequential):
        y_pred = [np.argmax(y_pred[i], axis=1) for i in (0,1)]

    confusao = [sklearn.metrics.confusion_matrix(y[i], y_pred[i]) for i in (0,1)]

    fig, ax = plt.subplots(1,2,figsize=(13,6.5))
    
    for i in (0,1):
        
        # dividindo cada valor da matriz pelo número total de imagens em cada classe
        row_sums = confusao[i].sum(axis=0, keepdims=True)
        confusao_normalizada = confusao[i] / row_sums

        # sumindo com a diagonal, pra podermos analizar só os erros:
        np.fill_diagonal(confusao_normalizada, 0)
        
        # plotando a matriz!
        ax[i].matshow(confusao_normalizada, cmap=plt.cm.Blues);
        
        # plotando os valores numéricos
        for j in range(len(confusao[i])):
            for k in range(len(confusao[i])):
                text = ax[i].text(k, j, confusao[i][j, k],
                               ha="center", va="center", color="w" if j!=k else "dodgerblue")
                
        # para exibir todos os números
        ax[i].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        ax[i].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        
        # título
        ax[i].set_title(f'{labels[i]}\nAcurácia: {sklearn.metrics.accuracy_score(y[i], y_pred[i]).round(4)}');
display_score(m)

def logistic(u): return 1/(1 + np.exp(-u))

x = np.linspace(-6,6)
plt.plot(x,x>0,'--',label='heaviside')
plt.plot(x,logistic(x), label='logística')
plt.axis('off')
plt.legend();

m = keras.models.Sequential()

m.add(keras.layers.Dense(200, input_shape = (784,), activation="relu"))
m.add(keras.layers.Dense(10, activation="softmax"))
m.summary()
m.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
%%time 

H = m.fit(X_treino, y_treino, 
          batch_size = 200, epochs = 10, 
          validation_data = (X_validacao, y_validacao));
pd.DataFrame(H.history).plot();
plt.xlabel('epoch');
m.evaluate(X_validacao, y_validacao);
display_score(m)
def logistic(u): 
    return 1/(1 + np.exp(-u))
def relu(u): 
    return np.where(u > 0, u, 0)
def leaky_relu(u, a = 0.1): 
    return np.where(u > 0, u, a*u)
def elu(u, a=1): 
    return np.where(u > 0, u, a*(np.exp(u)-1))
          
funcs = [logistic, relu, leaky_relu, elu]

formulas = ["$f(u) = (1+e^{-u})^{-1}$",
            "$f(u)=\max(0,u)$",
            "$f(u,a)=\max(au,u)$",
            "$f(u,a)=\max(ae^{u-1},u)$"]
    
fig, ax = plt.subplots(1,len(funcs), figsize=(15,3), sharey=True)
x = np.linspace(-4,4)
    
for i in range(len(funcs)):
    ax[i].plot(x,funcs[i](x));
    ax[i].grid('on');
    ax[i].set_title(funcs[i].__name__+'\n'+formulas[i])

m = keras.models.Sequential([
    
keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape=(28, 28, 1)),
keras.layers.MaxPooling2D(pool_size = (2,2), strides = 2),
    
keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
keras.layers.MaxPooling2D(pool_size = (2,2), strides = 2),
    
keras.layers.Flatten(),
keras.layers.Dense(units=64, activation="relu"),
    
keras.layers.Dropout(0.2),
    
keras.layers.Dense(units=10, activation="softmax")
    
])

m.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
m.summary()
%%time
H = m.fit(X_treino.reshape(-1,28,28,1), y_treino,
          batch_size=64, epochs=10,
          validation_data = (X_validacao.reshape(-1,28,28,1), 
                             y_validacao))
pd.DataFrame(H.history).plot();
plt.xlabel('epoch');
display_score(m)
# localização dos exemplos na matriz de dados 
y_validacao_pred = np.argmax(m.predict(X_validacao.reshape(-1,28,28,1)), axis=1)
loc = np.where(y_validacao!=y_validacao_pred)[0]

# selecionando os dígitos, já no formato de matriz
digitos = [X_validacao[i].reshape(28,28) for i in loc]

# criando figura do matplotlib
fig, ax = plt.subplots(max(1,len(loc)//20+1),20,figsize=(18,1.7))

# plotando!
[ax.ravel()[i].imshow(digitos[i], cmap = matplotlib.cm.binary, interpolation="nearest") for i in range(len(loc))]

# desligando os eixos de todos os dígitos
[ax.ravel()[i].axis('off') for i in range(len(ax.ravel()))];
from IPython.display import YouTubeVideo
YouTubeVideo("4Lo3tcrz8U0")
YouTubeVideo("mIpc2TaiZv0")
YouTubeVideo("UC1RVfG2AfA")
YouTubeVideo("L0Q6cboXyLY")
technocast_PATH = '/kaggle/input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/'

technocast_train_path = technocast_PATH + 'train/'
technocast_test_path = technocast_PATH + 'test/'
dir1 = technocast_train_path+'/ok_front/'
dir2 = technocast_train_path+'/def_front/'

img1 = plt.imread(dir1+random.choice(os.listdir(dir1)))
img2 = plt.imread(dir2+random.choice(os.listdir(dir2)))

fig, ax = plt.subplots(1,2)

ax[0].imshow(img1)
ax[0].axis('off')
ax[0].set_title('ok')

ax[1].imshow(img2)
ax[1].axis('off');
ax[1].set_title('defeituosa');
def make_generators():

    datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255,
                                                           validation_split = 0.1)
    
    train_generator = datagen.flow_from_directory(directory = technocast_train_path, 
                                                  batch_size = 32,
                                                  target_size = (300, 300),
                                                  color_mode = "grayscale",
                                                  class_mode = "binary",
                                                  classes = {"ok_front": 0, "def_front": 1},
                                                  shuffle = True,
                                                  seed = 0,
                                                  subset = "training")

    validation_generator = datagen.flow_from_directory(directory = technocast_train_path,
                                                       batch_size = 32,
                                                       target_size = (300, 300),
                                                       color_mode = "grayscale",
                                                       class_mode = "binary",
                                                       classes = {"ok_front": 0, "def_front": 1},
                                                       shuffle = True,
                                                       seed = 0,
                                                       subset = "validation")
    
    return train_generator, validation_generator
def visualizeImageBatch(datagen, title):
    
    '''
    Adaptado de:
    https://www.kaggle.com/tomythoven/casting-inspection-with-data-augmentation-cnn
    '''
    
    mapping_class = {0: "0 (ok)", 1: "1 (defeituosa)"}
    
    images, labels = next(iter(datagen))
    images = images.reshape(32, *(300,300))
    
    fig, axes = plt.subplots(4, 8, figsize=(13,6.5))

    for ax, img, label in zip(axes.flat, images, labels):
        ax.imshow(img, cmap = "gray")
        ax.axis("off")
        ax.set_title(mapping_class[label], size = 12)

    fig.tight_layout()
    fig.suptitle(title, size = 16, y = 1.05)
train_generator, validation_generator = make_generators()

visualizeImageBatch(train_generator, 'Exemplo de minilote de treino')
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)

n_test = sum([len(files) for r, d, files in os.walk(technocast_test_path)])

test_generator = test_datagen.flow_from_directory(directory = technocast_test_path,
                                                  batch_size = n_test,
                                                  target_size = (300, 300),
                                                  color_mode = "grayscale",
                                                  class_mode = "binary",
                                                  classes = {"ok_front": 0, "def_front": 1},
                                                  shuffle = False)
fig, ax = plt.subplots(1,3,figsize=(10,3))

sns.countplot(train_generator.classes,ax=ax[0])
sns.countplot(validation_generator.classes,ax=ax[1])
sns.countplot(test_generator.classes,ax=ax[2])

ax[0].set_title('Treino')
ax[1].set_title('Validação')
ax[2].set_title('Teste')

fig.suptitle('Proporção de classes (normais/defeituosas)')
fig.tight_layout(rect=[0, 0.03, 1, 0.92]);
def make_cnn():

    m = keras.models.Sequential([

    keras.layers.Conv2D(filters=16, kernel_size=(7,7), strides = 2, activation="relu", 
                        padding="same", input_shape=(300, 300, 1)),
    keras.layers.MaxPooling2D(pool_size = (2,2), strides = 2),
        
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D(pool_size = (2,2), strides = 2),
                
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D(pool_size = (2,2), strides = 2),
                
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D(pool_size = (2,2), strides = 2),
        
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D(pool_size = (2,2), strides = 2),
                
    keras.layers.Flatten(),
    keras.layers.Dense(units=64, activation="relu"),

    keras.layers.Dropout(0.2),

    keras.layers.Dense(units=1, activation="sigmoid")

    ])

    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return m
make_cnn().summary()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=5)
%%time

# número de redes no ensemble
n_models = 10

# listas que armazenarão os modelos e os históricos de resultados
m = [0]*n_models
H = [0]*n_models

for i in range(n_models):

    print(f'Rede {i+1}')
    print('----------------------------------------')
    
    # gerando conjuntos de treino/validaçao diferentes a cada treinamento
    train_generator, validation_generator = make_generators()
    
    # gerando o modelo
    m[i] = make_cnn()
    
    # checkpoint para salvar o melhor modelo para a rede em questão
    checkpoint = keras.callbacks.ModelCheckpoint(f"technocast_cnn_{i+1}.hdf5",
                                                 save_best_only = True,
                                                 monitor = "val_loss")
    
    print('Treinando...')
    
    # treinando a rede em questão
    H[i] = m[i].fit(train_generator, validation_data = validation_generator, 
                    epochs = 30, callbacks=[early_stop,checkpoint], verbose=0)
    
    # imprimindo resultados após o término do treino da rede em questão
    epoch_min = pd.DataFrame(H[i].history).idxmin(axis=0)['val_loss']
    print(f"Épocas: {len(H[i].history['loss'])}") 
    print(f"loss: {H[i].history['loss'][epoch_min]:.4}, accuracy: {H[i].history['accuracy'][epoch_min]:.4}")
    print(f"val_loss: {H[i].history['val_loss'][epoch_min]:.4}, val_accuracy: {H[i].history['val_accuracy'][epoch_min]:.4}")
    print('----------------------------------------\n')
fig, ax = plt.subplots(2,5, figsize=(20,5))

for i in range(n_models):
    
    pd.DataFrame(H[i].history).plot(ax=ax.ravel()[i], legend = True if i==0 else False);
    plt.xlabel('epoch');
best_models = [keras.models.load_model(f"technocast_cnn_{i+1}.hdf5") for i in range(n_models)]
def ensemble_prediction(generator):
    
    y_probs = [best_models[i].predict(generator).squeeze() for i in range(len(best_models))]
    return np.mean(np.array(y_probs), axis=0)
y_probs = ensemble_prediction(test_generator)
y = test_generator.classes
y_pred = y_probs>0.5

print(f'Acurácia:{sklearn.metrics.accuracy_score(y, y_pred):.4}')
print('------------------------------')
print('Matriz de confusão:')
print(sklearn.metrics.confusion_matrix(y, y_pred))
print(f'Precisão: {sklearn.metrics.precision_score(y, y_pred):.4}')
print(f'Revocação: {sklearn.metrics.recall_score(y, y_pred):.4}')
print(f'F1: {sklearn.metrics.f1_score(y, y_pred):.4}')
prob_x = np.linspace(0.01,0.99)

recall = [0]*len(prob_x)
f1 = [0]*len(prob_x)
precision = [0]*len(prob_x)

for i in range(len(prob_x)):
    y_pred_rp_curve = y_probs>prob_x[i]
    recall[i] = sklearn.metrics.recall_score(y, y_pred_rp_curve)
    f1[i] = sklearn.metrics.f1_score(y, y_pred_rp_curve)
    precision[i] = sklearn.metrics.precision_score(y, y_pred_rp_curve)

plt.plot(prob_x, recall,'.--')
plt.plot(prob_x, f1,'*-')
plt.plot(prob_x, precision,'.-')

plt.xlabel('Probabilidade de corte')
plt.legend(['revocação','$F_1$','precisão']);
direc = technocast_test_path
imgs = [plt.imread(direc+file) for file in test_generator.filenames]

fig, ax = plt.subplots(715//13,13, figsize=(18,100))

for i in range(715):
    
    ax.ravel()[i].imshow(imgs[i])
    ax.ravel()[i].axis('off');
    
    color = ('black' if ((test_generator.labels[i]==0 and y_probs[i]<0.5)) or 
                        (test_generator.labels[i]==1 and y_probs[i]>=0.5) 
             else 'red')
    
    ax.ravel()[i].set_title(f'{test_generator.labels[i]}\n {y_probs[i]:.2}', color=color)
'''
No caso de o modelo atingir acurácia de 100%, esta célula não rodará
'''
if sklearn.metrics.accuracy_score(y, y_pred)<1:

    direc = technocast_test_path

    # selecionando as imagens com predições erradas

    wrong_positions = np.where(y!=y_pred)[0]
    wrong_files = [test_generator.filenames[i] for i in wrong_positions]

    imgs = [plt.imread(direc+file) for file in wrong_files]

    # probabilidades para as imagens com predições erradas
    y_probs_wrong = ensemble_prediction((next(test_generator)[0][wrong_positions]))

    fig, ax = plt.subplots(1,len(wrong_positions), figsize = (20,5))

    # transformando objetos ax, y_probs_wrong e wrong_positions em containers caso nao sejam
    # para podermos entrar no loop abaixo mesmo quando há apenas uma posição errada
    ax = np.array([ax]) if not hasattr(type(ax), '__iter__') else ax
    y_probs_wrong = [y_probs_wrong] if not hasattr(type(y_probs_wrong), '__iter__') else y_probs_wrong
    wrong_positions = [wrong_positions] if not hasattr(type(wrong_positions), '__iter__')  else wrong_positions

    # plotando!

    for i in range(len(wrong_positions)):

        ax[i].imshow(imgs[i])
        ax[i].axis('off');
        ax[i].set_title(f'{test_generator.labels[wrong_positions[i]]}\n {y_probs_wrong[i]:.2}')
        ax[i].text(2, 15, str(wrong_positions[i]))

    fig.suptitle('Amostra(s) classificada(s) incorretamente', fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.9]);
'''
No caso de o modelo atingir acurácia de 100%, esta célula não rodará
'''
if sklearn.metrics.accuracy_score(y, y_pred)<1:
    
    direc = technocast_test_path

    # selecionando as imagens com predições longe de 0 e 1

    mask = np.logical_and(y_probs > 0.2, y_probs < 0.8)

    unusual_prob_positions = np.where(mask)[0]
    unusual_prob_files = [test_generator.filenames[i] for i in unusual_prob_positions]

    imgs = [plt.imread(direc+file) for file in unusual_prob_files]

    # predições propriamente ditas
    y_probs_unusual = ensemble_prediction(next(test_generator)[0][unusual_prob_positions])

    # gerando a janela do gráfico

    n_columns = 5
    n_lines = max(1,max(1,len(unusual_prob_positions)//n_columns+1))

    fig, ax = plt.subplots(n_lines,n_columns, 
                           figsize = (20,4*n_lines))

    # zerando todos os eixos antes de entrarmos no loop
    [ax.ravel()[i].axis('off') for i in range(len(ax.ravel()))]

    # plotando!

    for i in range(len(unusual_prob_positions)):

        ax.ravel()[i].imshow(imgs[i])

        color = ('black' if ((test_generator.labels[unusual_prob_positions[i]]==0 and y_probs_unusual[i]<0.5)) or 
                            (test_generator.labels[unusual_prob_positions[i]]==1 and y_probs_unusual[i]>=0.5) 
                 else 'red')

        ax.ravel()[i].set_title(f'{test_generator.labels[unusual_prob_positions[i]]}\n {y_probs_unusual[i]:.2}',
                               color = color)

        ax.ravel()[i].text(2, 15, str(unusual_prob_positions[i]))

    suptitle1 = f'{len(unusual_prob_positions)} amostras a serem encaminhadas para revisão humana ({(len(unusual_prob_positions)/715*100):.2}% do total)'
    suptitle2 = f'das quais {np.in1d(unusual_prob_positions, wrong_positions).sum()} classificada(s) incorretamente.'

    fig.suptitle(suptitle1+'\n'+suptitle2,fontsize=18);

    fig.tight_layout(rect=[0, 0.03, 1, 0.9])
