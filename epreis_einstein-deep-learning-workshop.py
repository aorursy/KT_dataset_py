# As imagens do Dataset "RxTorax" que contém 350 imagens normais e 350 de derrame já estão vinculadas a este "Notebook"
# Para editar o dataset vinculado a este kernel vá em "Add Data" no menu à direita. 
# Agora que ja temos o dataset pronto, vamos criar uma lista dos arquivos utilizando o glob, que lê e lista os arquivos que existem dentro de uma pasta

from glob import glob

derrame_dir = '../input/effusion/*/*.png' #Define o caminho das pastas que contém as imagens
normal_dir = '../input/normal/*/*.png'

derrame_lista = glob(derrame_dir) #Lista os arquvos dentro de cada uma das pastas, usando o glob()
normal_lista = glob(normal_dir)

print('Número de casos com derrame: ', len(derrame_lista)) #Visualiza o tamanho da lista
print('Número de casos normais: ', len(normal_lista))
print ('\nEtapa Concluída. Vamos para a próxima!')

#Execute com SHIFT + ENTER

#O resultado esperado é:
#Número de casos com derrame:  350
#Número de casos normais:  350
import cv2 #Para abrir o arquivo de imagem, utilizaremos o openCV, uma biblioteca aberta de visão computacional
from matplotlib import pyplot as plt #Biblioteca de plotagem de gráficos chamada a matplotlib

#Em 'Classe' digite 'N' para imagens da classe 'Normal', 
#ou substitua-o por 'D' para imagens com derrame
Classe = 'D'

#Escolha uma imagem entre 0 e 349:
ID_arquivo = 301

if Classe == 'N':
    classe = 'Normal'
    classe_lista = normal_lista
elif Classe == 'D':
    classe = 'Derrame'
    classe_lista = derrame_lista
    
imagem = cv2.imread(classe_lista[ID_arquivo])
plt.imshow(imagem)
plt.show()
    
print('Classe: ',classe)
print ('\nEtapa Concluída. Vamos para a próxima!')

#Execute com SHIFT + ENTER
#Pode modificar o ID_arquivo ou a Classe para vizualizar outras imagens. Pode repetir quantas vezes quiser
import numpy as np #biblioteca NumPy para trabalharmos com matrizes
# Por que usamos matrizes? A entrada de informações nas redes neurais se dá nesse formato,
# o computador enxerga as imagens dessa forma e permite processamento computacional paralelo e maior velocidade de processamento.

dataset = [] # cria uma lista vazia para incluir as imagens do dataset
labels = [] # cria uma lista vazia para incluir a categoria a qual cada imagem pertence (0 ou 1)

for arquivo in derrame_lista: # para cada arquivo de imagem na lista derrame:
    img = cv2.imread(arquivo, cv2.IMREAD_GRAYSCALE) #abre o arquivo em escala de cinzas
    img = cv2.resize(img, (256,256)) #redimensiona a imagem para 256 x 256
    dataset.append(img) #adiciona essa imagem na lista do dataset que criamos acima
    labels.append(1) #informa que ela é um caso de derrrame (1)

#Agora faremos o mesmo para as imagens sem derrame
for arquivo in normal_lista:
    img = cv2.imread(arquivo, cv2.IMREAD_GRAYSCALE) 
    img = cv2.resize(img, (256,256)) 
    dataset.append(img)
    labels.append(0) #mas agora informaremos que ela é um caso normal (0)
    
dataset = np.asarray(dataset, dtype=np.float32) #transforma a lista de variáveis numa matriz
labels = np.asarray(labels)

for i in range(len(dataset)):
  dataset[i] = (dataset[i] - np.average(dataset[i], axis= (0, 1))) / np.std(dataset[i], axis= (0, 1)) #faremos a normalização, dividindo a média pelo desvio padrão
  # normalização diminui a variabilizada do dataset, deixando os valores mais próximos um do outro
  
print('Dimensões da Matriz: ',dataset.shape)
print ('\nEtapa Concluída. Vamos para a próxima!')

#Vamos ver qual o tamanho dessa matriz 'dataset'
#Esperamos que a primeira dimensão dela seja de 700 (350 casos de derrame e 350 normais)
#A segunda e a terceira dimensões devem ser 256.

# a saída esperada é (700, 256, 256)
#Vamos separar nosso dataset em grupos de treinamento, validação e teste. Para isso, usaremos a biblioteca sklearn.
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#Vamos dividir o grupo de treino, validação e teste na proporção de cerca de 70%/15%/15%, 
#com valores aproximados para ficarmos com números redondos de 500/100/100

dataset_train, dataset_test, labels_train, labels_test = train_test_split(dataset[:,...,np.newaxis], labels[:,...,np.newaxis], test_size=0.142, random_state=80)
dataset_train, dataset_val, labels_train, labels_val = train_test_split(dataset_train, labels_train, test_size=0.1651, random_state=80)

print('(Número de imagens, Imagem_X, Imagem_Y, canais de cor) (Número de labels, 1)')
print(dataset_train.shape, labels_train.shape)
print(dataset_val.shape, labels_val.shape)
print(dataset_test.shape, labels_test.shape)

#Você deve ver a seguinte saída:
#(500, 256, 256, 1) (500,1)
#(100, 256, 256, 1) (100,1)
#(100, 256, 256, 1) (100,1)

print ('\nEtapa Concluída. Vamos para a próxima!')
#Vamos importar as função do Keras que faz data augmentation:

from keras.preprocessing.image import ImageDataGenerator

#Aqui podemos definir diferentes variáveis que vão definir como as imagens  
#vão mudar em rotação, "corte" ou zoom.

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

print ('\nEtapa Concluída. Vamos para a próxima!')
#Nesta etapa vamos criar a arquitetura da nossa rede neural convolucional, 
#Utilizaremos a biblioteca Keras, própria para Deep Learning em Python
#Inicialmente vamos importar as funções do Keras que iremos utilizar:

from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D, GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Input, Concatenate, add
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, LeakyReLU
from keras.utils.np_utils import to_categorical

#Agora criaremos a estrutura da rede neural convolucional

#A primeira camada define o tamanho da camada de entrada da rede
imgs = Input(shape=(256,256,1))

#Lembre que a nossa matriz com todas as 556 imagens de cada categoria tem o formato (556, 256, 256)
#Nesse caso a entrada (input) da rede é cada imagem individualmente
#Ou seja, uma imagem de tamanho 256 x 256 pixels e 1 canal de cor (escala de cinzas) (256, 256, 1)

#Agora vamos adicionar a primeira camada convolucional
x = Conv2D(8, 3, padding='same', activation='relu')(imgs)

#Em seguida, adicionamos uma camada MaxPool, que irá reduzir em 75% o tamanho da saída da camada convolucional.
#Fazemos isso para evitar que o número de parâmetros da rede aumente demais.
x = MaxPool2D()(x)

#Adicionaremos mais camadas convolucionais, seguidas de MaxPool
x = Conv2D(8, 3, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(12, 3, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(12, 3, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(20, 5, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(20, 5, padding='same', activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(50, 5, padding='same', activation='relu')(x)
x = GlobalAveragePooling2D()(x)

#Finalmente adicionaremos duas camadas densas, chamadas de 'Fully Connected Layers'.
#Essas camadas são redes neurais não convolucionais.
#Estas camadas recebem os parâmetros das primeiras e ajudam a realizar a classificação dos resultados
x = Dense(128, activation='relu')(x)

#Dropout é uma técnica para reduzir overfitting onde excluímos parte dos neurônios de uma camada.
x = Dropout(0.6)(x)

x = Dense(32, activation='relu')(x)

#Nossa camada de "output" tem o argumento "1" pois a saída da rede é a classificação derrame x não-derrame
#Ou seja, a saída da rede é apenas um número (0 ou 1)
outputs = Dense(1, activation='sigmoid')(x)

inputs = imgs

#Por fim, definiremos nossa rede com a entrada e a saída da rede
RadEinstein_CNN = Model(inputs=inputs, outputs=outputs)

#Agora, definiremos o método de otimização da rede: ADAM, com a taxa de aprendizado e de decaimento
#Cada um desses parâmetros é ajustável.
custom_adam = optimizers.Adam(lr=0.0005, decay=0.0002)

#Compila o modelo definindo o tipo de função 'loss', otimização e a métrica.
RadEinstein_CNN.compile(loss='binary_crossentropy', optimizer=custom_adam, metrics=['acc'])

print('Veja abaixo as camadas da sua rede neural' )
print('Note que cada camada contém uma quantidade diferente de parâmetros a serem treinados')
print('No final da lista, note que nossa rede contém um total de 54.627 parâmetros treináveis')
print ('\nRadEinstein_CNN Sumary')
RadEinstein_CNN.summary()
print ('\nEtapa Concluída. Vá para a próxima!')
import time #Função time para medirmos o tempo de treinamento

checkpointer = ModelCheckpoint(filepath='Melhor_modelo.hdf5', monitor='val_loss',
                               verbose=1, save_best_only=True) #Salvar o melhor modelo que for encontrado durante o treino

print('Treinando a Rede RadEinstein_CNN:')

Valida = (dataset_val, labels_val)

#Finalmente vamos treinar a nossa rede
#Se quiser treinar sua rede um pouco mais, altere o número de epochs, e vamos ver os diferentes resultados.

start_time = time.time()
hist = RadEinstein_CNN.fit_generator(datagen.flow(dataset_train, labels_train, batch_size=16), steps_per_epoch=len(dataset_train), epochs=8, 
                    validation_data= (dataset_val, labels_val), 
                    callbacks=[checkpointer])

#Definimos o treinamento com o dataset de treino, realizando validação no dataset de validação.
#O treinamento não usa o dataset de teste, ficará guardado para avaliarmos nossa rede depois.

tempo = str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
print('')
print('Treino finalizado em %s' % tempo)

#Por fim, plotamos os resultados de evolução da medida de erro (loss) e acurácia ao longo dos epochs
plt.plot(hist.history['loss'], 'b-', label='train loss')
plt.plot(hist.history['val_loss'], 'r-', label='val loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

plt.plot(hist.history['acc'], 'b-', label='train accuracy')
plt.plot(hist.history['val_acc'], 'r-', label='val accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.show()

print ('\nAgora vamos avaliar o modelo no dataset de teste. Vamos para a próximo comando.')

#Execute esse código com SHIFT + ENTER
from keras.models import load_model #Vamos importar a função do keras que abre modelos salvos previamente

melhor_modelo = load_model('Melhor_modelo.hdf5') #Abrimos o melhor modelo que salvamos no treinamento

print ('\nPesos da rede neural atualizados para os da melhor época.')
print ('\nEtapa Concluída. Vamos para a próxima!')
#Execute esse código com SHIFT + ENTER
#Usamos a função evaluate para avaliar a acurácia do nosso modelo no grupo de teste
print('Acurácia no grupo de teste: ', melhor_modelo.evaluate(dataset_test, labels_test, verbose=0)[1])

print ('\033[1m' + 'Etapa Concluída. Vamos para a próxima!')
#Execute esse código com SHIFT + ENTER
# Instale a biblioteca Keras-vis para visualizarmos como a rede neural enxerga a imagem.
# Se tiver problemas com a instalação ative o campo "internet" na barra lateral de configurações -->
!pip install -q keras-vis
print ('\033[1m' + 'Instalação concluída. Vamos para a próxima etapa!')

#Agora vamos fazer a inferência e visualizar o mapa de ativação das imagens do nosso dataset de teste.

from vis.visualization import saliency
from vis.visualization import activation_maximization
from vis.visualization.__init__ import get_num_filters

ID_imagem = 14
Layer = -17

for i in range(10):
    #Escolha uma imagem no grupo de Teste, de 0 a 99:
    ID_imagem = i
    
    img = dataset_test[ID_imagem]

    plt.imshow(np.squeeze(img), cmap='gray')#Mostra a imagem que escolhemos
    plt.show()
    
    heatmap = saliency.visualize_cam(melhor_modelo, Layer, filter_indices=range(get_num_filters(melhor_modelo.layers[Layer])), seed_input=img, penultimate_layer_idx=Layer-1, \
    backprop_modifier=None, grad_modifier=None)
    plt.imshow(np.squeeze(img), cmap='gray')
    plt.imshow(heatmap,alpha=0.5)
    plt.show()

    #Vamos mostrar a qual classe ela pertence
    print('Classe:', 'normal' if labels_test[ID_imagem]==0 else 'derrame')

    predicao = np.round(melhor_modelo.predict(dataset_test[ID_imagem][np.newaxis,:,...], verbose=0))==0
    print('Predição:', 'normal' if predicao else 'derrame')


from IPython.display import HTML, display
display(HTML('<img src="https://media0.giphy.com/media/S6TEoUBJuGfQCoGl8l/giphy.gif?cid=ecf05e47oxgcxjd98g6ghdaalezi4jhrkxmig00jf8vhg2np&rid=giphy.gif">'))

print ('\n' + '\033[1m' + 'Etapa Concluída. Parabéns, você finalizou o Notebook!')


#Execute esse código com SHIFT + ENTER
