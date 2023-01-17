# Veamos la data que tenemos
!ls ../input
!ls -U ../input/train/ | tail -n 5
!ls ../input/test/ | tail -n 5
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm_notebook
from pathlib import Path
%matplotlib inline
train_path = Path('../input/train/')
cat_imgs, dog_imgs = [], []
for e in train_path.iterdir():
    if 'cat' in e.name: cat_imgs.append(e)
    else              : dog_imgs.append(e)
len(cat_imgs), len(dog_imgs)
# Definimos una funcion para leer una imagen y hacer el preprocesamiento
from keras.applications.resnet50 import preprocess_input
img_size = 224

def read_img(path):
    x = Image.open(path)
    x = x.resize((img_size, img_size))
    x = np.asarray(x, np.float32)
    return preprocess_input(x)
# Cargamos una muestra de imagenes para train y validation

# Primero inicializamos los arrays que vamos a usar
x_train = np.ndarray(shape=(2000, img_size, img_size, 3), dtype=np.float32)
y_train = np.zeros(shape=(2000), dtype=np.int8)
x_val = np.ndarray(shape=(200, img_size, img_size, 3), dtype=np.float32)
y_val = np.zeros(shape=(200), dtype=np.int8)
# Cargamos el train set
for i,e in tqdm_notebook(enumerate(cat_imgs[:1000] + dog_imgs[:1000])):
    x_train[i] = read_img(e)

y_train[1000:] = 1 # cat -> 0 | dog -> 1
# Cargamos el validation set
for i,e in tqdm_notebook(enumerate(cat_imgs[1000:1100] + dog_imgs[1000:1100])):
    x_val[i] = read_img(e)

y_val[100:] = 1 # cat -> 0 | dog -> 1
from keras.applications.resnet50 import ResNet50
base_model = ResNet50(include_top=False, input_shape=(img_size,img_size,3), pooling='avg')
base_model.summary()
base_model.input, base_model.output
# Dado que solo queremos entrenar las capas densas del modelo que agregaremos
# en el siguiente paso, vamos a setear "trainable = False" para que los pesos
# de la red entrenada no cambien.
base_model.trainable = False
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

top_model = Sequential([
    Dense(128, activation='relu', input_shape=(2048,)),
    Dense(1, activation='sigmoid')
])

top_model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
top_model.summary()
final_model = Sequential([base_model, top_model])

final_model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
final_model.summary()
# log = final_model.fit(x_train, y_train, batch_size=64, validation_data=[x_val, y_val])
precomputed_train = base_model.predict(x_train, batch_size=128, verbose=1)
precomputed_train.shape
precomputed_val = base_model.predict(x_val, batch_size=128, verbose=1)
precomputed_val.shape
# Ahora podemos usar un batch_size mas grande, ya que los features son mas pequeños
# que las imagenes.
log = top_model.fit(precomputed_train, y_train, epochs=5, batch_size=256, validation_data=[precomputed_val, y_val])
def show_results(log):
    fig, axes = plt.subplots(1, 2, figsize=(14,4))
    ax1, ax2 = axes
    ax1.plot(log.history['loss'], label='train')
    ax1.plot(log.history['val_loss'], label='validation')
    ax1.set_xlabel('epoch'); ax1.set_ylabel('loss')
    ax2.plot(log.history['acc'], label='train')
    ax2.plot(log.history['val_acc'], label='validation')
    ax2.set_xlabel('epoch'); ax2.set_ylabel('accuracy')
    for ax in axes: ax.legend()
show_results(log)
test_path = Path('../input/test/')
test_files = list(test_path.iterdir())
def get_class(path):
    # Cargar la imagen del path
    img = Image.open(path)
    
    # Cambiar el tamaño de la imagen
    img_resized = img.resize((224, 224))
    
    # Cambiar a formato numpy y preprocesar
    x = np.asarray(img_resized, np.float32)[None]
    x = preprocess_input(x)
    
    # Obtener predicciones
    y = final_model.predict(x)
    
    # Decodear predicciones
    pred = 'cat' if y < 0.5 else 'dog'
    
    # Mostrar la imagen
    plt.imshow(img)
    plt.axis('off')
    plt.title(pred, size=14)
    
    return
sample = np.random.choice(test_files)
get_class(sample)

