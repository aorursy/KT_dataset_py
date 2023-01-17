import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, sys # modulos do sistema

from time import time
# import lbrary for image manipulation
import matplotlib.pyplot as plt   # biblioteca gráfica para gráficos
import matplotlib.image as mpimg  # biblioteca gráfica para gráficos
import cv2                        # library for image manipulation

# bibliotecas de IA
from skimage.transform import resize
from skimage.io import imsave
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

print(os.listdir("../input/weights/"))
print(os.listdir('../input/airbus-ship-detection/'))
# lendo o arquivo que contem todas as imagens de treinamento e as informações das máscara
tss = pd.read_csv('../input/airbus-ship-detection/train_ship_segmentations_v2.csv')
tss.keys()
tss.shape
# PARAMETROS

# teste
# comeco = 0
# fim = 10

# ignorado
# comeco = 230000
# fim = 270000

# 11 treinanemtno com 10000
comeco = 40000
fim = 80000

EPS = 10
BS = 32

pesos = 'train-12.h5'

total = fim-comeco #65515(50%) # 31030(23%) # len(tss['ImageId']) # 131030

image_rows = 768
image_cols = 768

img_rows = 192
img_cols = 192
smooth = 1.

dim = 589824 # dimensão maxima da máscara 1d
print('total de imagens', total)
# ------------------------------------------------ #
# -------- CRIANDO DADOS DE TREINAMENTO ---------- #
# ------------------------------------------------ #

# selecionar o nome da imagem e ler ela com o cv2, 
# após obter a marcação e gerar a mascara 
# adicionar a imagem e mascara no vertor de array para treinamento

# criando todas as mascaras das imagens
def create_train_data():
    
    print(''*30)
    print('criando dados de treinamento')
    print('-'*30)
    
    imgs = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    
    a = 0 # indice
    
    # controlar qais valores do dataframe eu vou ler
    for i in range(comeco, fim):

        img = cv2.imread('../input/airbus-ship-detection/train_v2/'+tss.iloc[i]['ImageId'], 0) # lendo a imagem em tons de cinza 
        
        str_mask = tss.iloc[i]['EncodedPixels'] # quando a imagem não possuir embarcação, a mesma passa pra outra 

        img_1d = np.zeros((dim),dtype=np.uint8) # cria um vetor 1D 
        
        if isinstance(str_mask, str): # se for string ela seta a mascara como branca

            str_mask = str_mask.split(' ') # obtem os pares de informações, posição e quantidade de pixels brancos
            
            for i in range(1,len(str_mask),2): # setar os valores da máscara
                for j in range(int(str_mask[i])+1): 
                    position = int(str_mask[i-1])
                    if position+j < dim: # assim não acessa posições fora do tamanho do array
                        img_1d[position+j] = 255

        img_2d = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8) # cria a imagem 2D de fato

        
        indice = 0
        for j in range(img.shape[1]): # seta na imagem 2D os valores da imagem 1D
            for i in range(img.shape[0]):
                img_2d[i][j] = img_1d[indice]
                indice += 1
        
        # mas antes as imagens são convetidas em arrays        
        img = np.array([resize(img, (img_cols, img_rows), preserve_range=True)])
        img_mask = np.array([resize(img_2d, (img_cols, img_rows), preserve_range=True)])

        # as imagens são setadas nos vetores de imagens
        imgs[a] = img
        imgs_mask[a] = img_mask
        
        if a % 100 == 0:
            logg = 'Done: {0}/{1} images'.format(a, total) 
            sys.stdout.write('\r'+logg)
            #print('Done: {0}/{1} images'.format(a, total))
            
        a += 1
    return imgs, imgs_mask
def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = imgs[i]
        
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p
inicio = time()
# criar os dados de treinamento
x_train, y_train = create_train_data()

print(x_train.shape, ' y ', x_train.shape)

print('pre-processing')
x_train = preprocess(x_train)
y_train = preprocess(y_train)
print('.')

print('convert tensor to float32')
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
print('.')

print('normalizando as imagens')
x_train = x_train / 255.
y_train = y_train / 255.
print('.')

for i in range(5):
    
    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    
    axs[0].imshow(x_train[i,:,:,0], cmap='gray')
    
    axs[1].imshow(y_train[i,:,:,0], cmap='gray')
    
plt.suptitle("Examples of Images and their Masks")
# ------------------------------------------------ #
# ------------ ARQUITETURA DA REDE --------------- #
# ------------------------------------------------ #

def get_unet(entrada, weights_path=None):
    
    print('entrada da rede = ', entrada)
    
    inputs = Input(entrada) # camada de entrada, resolução de imagens em um canal
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    if weights_path:
        model.load_weights(weights_path)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

# metricas para avaliar a perda da segmetnação durate o treinamento
smooth = 1.

def dice_coef(y_true, y_pred): 
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
model = get_unet(x_train[0].shape,'../input/weights/'+pesos) # carrega os pesos aqui do treinamento anterior, se tiver e ou for melhor
model.summary()
model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

print('-'*30)
print('Fitting model...')
print('-'*30)
log = model.fit(x_train, y_train, batch_size=BS, epochs=EPS, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])
log.history.keys()
print('val_loss ........: ', log.history['val_loss'][len(log.history['val_loss'])-1])
print('loss ............: ', log.history['loss'][len(log.history['loss'])-1])
print('val_dice_coef....: ', log.history['val_dice_coef'][len(log.history['val_dice_coef'])-1])
print('dice_coef .......: ', log.history['dice_coef'][len(log.history['dice_coef'])-1])
fim = time()

# tempo em segundos
gasto = fim - inicio

# tempo em minutos
gasto /= 60

# tempo em horas
horas = gasto / 60

titulo = " Model Loss - Treinamento "+str(pesos)+"  \n "+str(total)+" imagens . "+str(EPS)+" epocas . "+str(BS)+" batch szie \n Tempo Gasto : %.2f minutos ou %.2f horas " %(gasto, horas)+" \n "+label_val_loss+" and "+label_loss
label_val_loss = 'val_loss %.2f '  %(log.history['val_loss'][len(log.history['val_loss'])-1])
label_loss = 'loss %.2f ' %(log.history['loss'][len(log.history['loss'])-1])

nome_fig = str(pesos)+'_.png'

# ---

plt.plot(log.history['val_loss'], '--go', label='val_loss')
plt.plot(log.history['loss'], '--ro', label='loss')

# plt.plot(log.history['val_dice_coef'], '--bo', label='val_dice_coef')
# plt.plot(log.history['dice_coef'], '--yo', label='dice_coef')



plt.title(titulo)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper center', shadow=True, fontsize='x-large',  bbox_to_anchor=(1.25, 0.7), ncol=1)
plt.savefig(nome_fig)


