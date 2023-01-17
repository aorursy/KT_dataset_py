! cp ../input/my-python/* ../working/
# General imports
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
%matplotlib inline
from tqdm import tqdm_notebook
import gc
from IPython.display import FileLink, FileLinks

# Sklearn imports
from sklearn.model_selection import train_test_split

# Keras imports
from keras.applications import InceptionV3, VGG19, ResNet50, Xception
from keras.applications.inception_v3 import preprocess_input
#from keras.applications.resnet50 import preprocess_input
#from keras.applications.xception import preprocess_input
from keras.models import Sequential, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint

# Image processing and augmentation
import imgaug as ia
from imgaug import augmenters as iaa
import cv2

# Cyclic learning rates
from clr import LRFinder
from clr_callback import CyclicLR
# General constants
DATA_DIR = '../input/geotagged-photos/'
PHOTO_DIR = DATA_DIR + 'geotagged photos from farmers 2018/'
SEED = 2001
BATCH_SIZE = 32
INPUT_SIZE = 299
# Leer la información sobre las fotografías. Vemos que todas las entradas tienen asociada una fotografía
photos = pd.read_excel(os.path.join(DATA_DIR, 'GEOTAGGED_PHOTOS FROM FARMERS.xlsx'))
photos['WITH_PHOTO'] = photos['PHOTO NAME'].map(lambda x: os.path.isfile(os.path.join(PHOTO_DIR, x)))
photos['WITH_PHOTO'].value_counts()
# Pero hay menos fotos que entradas: algunas fotografías tienen varias entradas asociadas (clasificación multilabel)
print('Unique photo names', len(photos['PHOTO NAME'].unique()))
# Veamos cuántas entradas hay de cada tipo de producto
photos['PRODUCT NAME'].value_counts()
# Veamos algunas fotografías aleatorias

def mostrar_fotos(df, filas, columnas, photo_column='PHOTO NAME'):
    fig, axs = plt.subplots(filas, columnas, figsize=(5 * filas, 5 * columnas), squeeze=True)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    axs = axs.reshape(-1)
    indices = np.random.choice(df.shape[0], filas * columnas, replace=False)
    for ax, i in zip(axs, indices):
        img = cv2.imread(os.path.join(PHOTO_DIR, df.iloc[i][photo_column]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)
mostrar_fotos(photos, 5, 5)
erroneos = pd.read_csv(DATA_DIR + 'erroneos.txt', delim_whitespace=True, usecols=[3], names=['photo'])
mostrar_fotos(erroneos, 5, 5, photo_column='photo')
photos['ERRONEA'] = photos['PHOTO NAME'].apply(lambda x: x in erroneos.photo.values)
photos['ERRONEA'].value_counts()
# Excluimos las erróneas
photos = photos[photos.ERRONEA == False]
# Vamos a cambiar los productos para simplificar, mapeando a un único producto todos cereales, los pastos y también todos los barbechos
CLASSES = ['CEREAL', 'GRASSLAND', 'FALLOW']
NUM_CLASSES = len(CLASSES)

single_product = {'PERMANENT GRASSLAND':'GRASSLAND', 'GRASSLAND':'GRASSLAND', 'EFA LYING FALLOW': 'FALLOW', 'NON-EFA LYING FALLOW':'FALLOW',
                 'BARLEY':'CEREAL', 'SOFT WHEAT': 'CEREAL', 'OATS': 'CEREAL', 'SORGHUM':'CEREAL'}
def mapeo(p):
    mp = single_product.get(p)
    if mp is None:
        return p
    else:
        return mp
    
photos['PRODUCTO'] = photos['PRODUCT NAME'].map(mapeo)
photos = photos[photos.PRODUCTO.isin(CLASSES)]
# Creamos un nuevo data frame sólo con las fotografías y las clases de interés
temp_df = photos.groupby(['PHOTO NAME','PRODUCTO'])['PRODUCTO'].count()
temp_df = temp_df.map(lambda x: 1 if x > 1 else x)
temp_df = temp_df.unstack().fillna(0).astype(np.int8)
temp_df.reset_index(drop=False, inplace=True)
temp_df.columns = ['PHOTO NAME'] + CLASSES
temp_df.head(20)
# Creamos una nueva columna para estratificar el conjunto de prueba y el de validación por producto
temp_df['stratify'] = temp_df.apply(lambda r: str(r[1]) + str(r[2]) + str(r[3]), axis='columns')
temp_df['stratify'].value_counts()
# Ya que el número de fotografías es limitado, vamos a cargarlas en memoria para acelerar el entrenamiento

num_fotos = temp_df.shape[0]

X = np.zeros(shape=(num_fotos, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.int16)
for idx in tqdm_notebook(range(num_fotos)):
    img = load_img(os.path.join(PHOTO_DIR, temp_df.iloc[idx]['PHOTO NAME']), target_size=(INPUT_SIZE, INPUT_SIZE))
    img = img_to_array(img)
    X[idx] = img
    
y = temp_df[temp_df.columns[1:-1]].values

gc.collect()
# Dividir las fotografías en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, stratify=temp_df['stratify'], random_state=SEED)
print (X_train.shape, X_val.shape, y_train.shape, y_val.shape)
augs = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Sometimes(0.2, iaa.Affine(rotate=(-20,20), mode='reflect')),  
    iaa.SomeOf((0,4), [
        iaa.AdditiveGaussianNoise(scale=0.01*255),        
        iaa.Sharpen(alpha=(0.0,0.3)),
        iaa.ContrastNormalization((0.8,1.2)),
        iaa.AverageBlur(k=(2,11)),
        iaa.Multiply((0.8,1.2)),
        iaa.Add((-20,20), per_channel=0.5),
        iaa.Grayscale(alpha=(0.0,1.0))
    ])
]) 

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                  rotation_range=30, 
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  width_shift_range=0.25,
                                  height_shift_range=0.25,
                                  zoom_range=0.3,
                                  brightness_range=(0.7,1.5),
                                  fill_mode='reflect')

val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)

val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False, seed=SEED)

def create_model(lr=0.0001):
  base = InceptionV3(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3), pooling='avg')
  #base = VGG19(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3), pooling='avg')
  #base = ResNet50(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3), pooling='avg')
  #base = Xception(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3), pooling='avg')
  
  for layer in base.layers:
        layer.trainable = False
    
  model = Sequential()
  model.add(base)
  model.add(Dense(512, activation='relu'))
  model.add(Dense(NUM_CLASSES, activation='sigmoid'))

  adam = Adam(lr=lr)
  sgd = SGD(lr=0.1, momentum=0.90, nesterov=True)
  model.compile(optimizer=sgd , loss='binary_crossentropy', metrics=['acc'],  )

  return model
# Run a lr range test to find good learning rate margins

model = create_model()

STEP_SIZE_TRAIN = X_train.shape[0] // BATCH_SIZE
STEP_SIZE_VALID = X_val.shape[0] // BATCH_SIZE

train_generator.reset()
val_generator.reset()

EPOCHS = 1
base_lr=0.0001
max_lr=10
step_size = EPOCHS * STEP_SIZE_TRAIN 
lrf = LRFinder(X_train.shape[0], BATCH_SIZE,
                       base_lr, max_lr,
                       # validation_data=(X_val, Yb_val),
                       lr_scale='exp', save_dir='./lr_find/', verbose=False)

history = model.fit_generator(train_generator, 
                              epochs=EPOCHS, 
                              steps_per_epoch=STEP_SIZE_TRAIN
                              ,validation_data=val_generator,
                              validation_steps=STEP_SIZE_VALID,
                              callbacks=[lrf]
                             )
fig = plt.figure(figsize=(15,7))
lrf.plot_schedule(clip_beginning=10)
model = create_model()
model.summary()
EPOCHS=60
STEP_SIZE_TRAIN = X_train.shape[0] // BATCH_SIZE
STEP_SIZE_VALID = X_val.shape[0] // BATCH_SIZE

train_generator.reset()
val_generator.reset()

clr = CyclicLR(base_lr=0.004, max_lr=0.02, step_size=2*STEP_SIZE_TRAIN, mode='exp_range')
checkpoint = ModelCheckpoint('land_use_predict_inception_v3.h5', monitor='val_loss', save_best_only=True, save_weights_only=False)

history = model.fit_generator(train_generator, 
                              epochs=EPOCHS, 
                              steps_per_epoch=3 * STEP_SIZE_TRAIN,
                              validation_data=val_generator,
                              validation_steps=STEP_SIZE_VALID,
                              callbacks=[clr, checkpoint]
                             )
def plt_history(history, metric, title, ax, val=True):
    ax.plot(history[metric])
    if val:
        ax.plot(history['val_' + metric])
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    
hist = history.history
fig, ax = plt.subplots(1,2, figsize=(15,6))
plt_history(hist, 'loss', 'LOSS', ax[0])
plt_history(hist, 'acc', 'ACCURACY', ax[1])
plt.savefig('history')
# Mostrar aleatoriamente imágenes y sus predicciones
def prediccion_aleatoria(X, y):
    idx = np.random.choice(range(X.shape[0]), 1)
    img = X[idx].squeeze()
    x = img_to_array(img)
    y_true = y[idx].squeeze()
    x = np.expand_dims(img, 0)
    x = preprocess_input(x)
    y_pred = model.predict_proba(x).squeeze()
    
    print('### REAL ###')
    print('CEREAL:{:2d} PASTO:{:2d} BARBECHO:{:2d}'.format(y_true[0], y_true[1], y_true[2]))
    print()
    print('### PREDICCIÓN ###')
    print('CEREAL:{:.2f} PASTO:{:.2f} BARBECHO:{:.2f}'.format(y_pred[0], y_pred[1], y_pred[2]))
    
    fig, ax = plt.subplots(1,1, figsize=(8,8), squeeze=True)
    ax.set_xticks([])
    ax.set_yticks([])
    
    
    ax.imshow(img)
# Recuperamos el mejor modelo salvado para hacer las predicciones
model = load_model('land_use_predict_inception_v3.h5')
prediccion_aleatoria(X_val, y_val)
prediccion_aleatoria(X_val, y_val)
prediccion_aleatoria(X_val, y_val)
prediccion_aleatoria(X_val, y_val)
prediccion_aleatoria(X_val, y_val)
prediccion_aleatoria(X_val, y_val)