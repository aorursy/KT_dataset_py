import math, keras, bcolz
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm_notebook
from pathlib import Path
%matplotlib inline
np.random.seed(34)
train_path = Path('../input/train/')
cat_imgs, dog_imgs = [], []
for e in train_path.iterdir():
    if 'cat' in e.name: cat_imgs.append(e)
    else              : dog_imgs.append(e)
        
# Hacemos una permutacion de los archivos para que esten en desorden
cat_imgs, dog_imgs = np.random.permutation(cat_imgs).tolist(), np.random.permutation(dog_imgs).tolist()
n_cat, n_dog = len(cat_imgs), len(dog_imgs)
n_cat, n_dog
# Vamos a usar 5000 imagenes para el validation set
n_val = 5000
n_train = n_cat + n_dog - n_val
train_files = cat_imgs[:-(n_val//2)] + dog_imgs[:-(n_val//2)]
val_files = cat_imgs[-(n_val//2):] + dog_imgs[-(n_val//2):]
# Definimos una funcion para leer una imagen y hacer el preprocesamiento
from keras.applications.resnet50 import preprocess_input
img_size = 224

def read_img(path):
    x = Image.open(path)
    x = x.resize((img_size, img_size))
    x = np.asarray(x, np.float32)
    return x
from keras.applications.resnet50 import ResNet50
from keras.models import Model

base_model = ResNet50(include_top=False, input_shape=(img_size,img_size,3))
base_model = Model(base_model.input, base_model.layers[-2].output)
base_model.trainable = False
base_model.summary()
base_model.input, base_model.output
from keras.models import Sequential
from keras.layers import GlobalAvgPool2D, Input, Conv2D, BatchNormalization, Activation
from keras.optimizers import Adam

m_in = Input((7, 7, 2048))
x = Conv2D(1024, 1, padding='same', activation='relu', kernel_initializer='he_uniform', use_bias=False)(m_in)
x = BatchNormalization()(x)
x = Conv2D(2, 1, padding='same')(x)
x = GlobalAvgPool2D()(x)
x = Activation('softmax')(x)
top_model = Model(m_in, x)

top_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
top_model.summary()
final_model = Sequential([base_model, top_model])
final_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
final_model.summary()
class DataSequence(keras.utils.Sequence):
    def __init__(self, files, batch_size):
        self.files = files
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.files) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.ndarray((self.batch_size, img_size, img_size, 3), np.float32)
        for i,f in enumerate(batch): batch_x[i] = read_img(f)
        return preprocess_input(batch_x)
train_seq = DataSequence(train_files, 100)
val_seq = DataSequence(val_files, 100)
precomputed_train = bcolz.carray(np.zeros((n_train, 7, 7, 2048), np.float32), chunklen=1, mode='w', rootdir='tmp_train')
precomputed_val = bcolz.carray(np.zeros((n_val, 7, 7, 2048), np.float32), chunklen=1, mode='w', rootdir='tmp_val')
y_train = np.zeros((n_train), np.int8)
y_val = np.zeros((n_val), np.int8)
y_train[n_train//2:] = 1
y_val[n_val//2:] = 1
for i,batch in tqdm_notebook(enumerate(train_seq), total=len(train_seq)):
    precomputed_train[i*100:(i+1)*100] = base_model.predict_on_batch(batch)
    if i == len(train_seq): break
for i,batch in tqdm_notebook(enumerate(val_seq), total=len(val_seq)):
    precomputed_val[i*100:(i+1)*100] = base_model.predict_on_batch(batch)
    if i == len(val_seq): break    
# Ahora podemos usar un batch_size mas grande, ya que los features son mas pequeños
# que las imagenes.
log = top_model.fit(precomputed_train, y_train, epochs=10, batch_size=256, validation_data=[precomputed_val, y_val])
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
    pred = 'cat' if np.argmax(y) == 0 else 'dog'
    
    # Mostrar la imagen
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'pred = {pred}', size=14)
    plt.show()
    
    return
for _ in range(3):
    sample = np.random.choice(test_files)
    get_class(sample)
preds_val = top_model.predict(precomputed_val, batch_size=256, verbose=1)
preds_val.shape
pred_classes = np.argmax(preds_val, axis=1)
idxs = np.where(y_val != pred_classes)[0]
errors = np.min(preds_val[idxs], axis=1)
idxs = idxs[np.argsort(errors)]
fig, axes = plt.subplots(3, 4, figsize=(18,12))
for ax, idx in zip(axes.flatten(), idxs):
    ax.imshow(Image.open(val_files[idx]))
    ax.set_title(f'''real label = {'dog' if idx > n_val/2 else 'cat'}
pred label = {'dog' if pred_classes[idx] == 1 else 'cat'}
error = {1 - np.min(preds_val[idx]):.4f}''')
    ax.axis('off')
plt.tight_layout()
# Vamos a crear un modelo para capturar una capa intermedia, ademas del output
map_model = Model(top_model.layers[0].input, [top_model.layers[-3].output, top_model.layers[-1].output])
import cv2

def get_feature_map(path):
    img = Image.open(path)
    img_resized = img.resize((224, 224))
    x = np.asarray(img_resized, np.float32)[None]
    x = preprocess_input(x)
    precomputed = base_model.predict(x)
    fmap, y = map_model.predict(precomputed)
    pred = np.argmax(y)
    fmap = cv2.resize(fmap[0,:,:,pred], img.size)
    fig, axes = plt.subplots(1, 2, figsize=(18,10))
    for ax in axes:
        ax.imshow(img)
        ax.axis('off')
        title = f"pred = {'cat' if pred == 0 else 'dog'}"
        if sample.name[:3] == 'cat' or sample.name[:3] == 'dog':
            title += f'\nreal = {sample.name[:3]}'
        ax.set_title(title, size=14)

    axes[0].imshow(fmap, cmap=plt.cm.RdGy_r, alpha=0.75)
    plt.show()
    
    return
for _ in range(5):
    sample = np.random.choice(val_files)
    get_feature_map(sample)
for i in np.random.permutation(idxs)[:5]:
    sample = val_files[i]
    get_feature_map(sample)
# Obtener resultados del test set
import pandas as pd

test_path = Path('../input/test/')
test_files = list(test_path.iterdir())

class TestDataSequence(DataSequence):
    def __getitem__(self, idx):
        batch = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.ndarray((self.batch_size, img_size, img_size, 3), np.float32)
        batch_ids = np.zeros((self.batch_size), np.int16)
        for i,f in enumerate(batch):
            batch_x[i] = read_img(f)
            batch_ids[i] = int(f.stem)
        return preprocess_input(batch_x), batch_ids

test_seq = TestDataSequence(test_files, 250)
preds, ids = [], []

for i,batch in tqdm_notebook(enumerate(test_seq), total=len(test_seq)):
    y_ = final_model.predict_on_batch(batch[0])
    preds += np.argmax(y_, axis=1).tolist()
    ids += batch[1].tolist()
    if i == len(test_seq): break    
results = pd.DataFrame({'id': ids, 'label': preds}).sort_values('id').drop_duplicates()
results.head()
results.to_csv('submission.csv', index=False)
from IPython.display import FileLink
FileLink('submission.csv')
# Eliminamos los archivos temporales del kernel
!rm -r tmp_train
!rm -r tmp_val