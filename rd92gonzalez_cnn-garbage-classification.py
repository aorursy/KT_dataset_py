import re

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

def list_dataset():

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))



            

# Add class name prefix to each path based on class name include in filename

def add_class_name_prefix(df, col_name):

    df[col_name] = df[col_name].apply(lambda x: x[:re.search("\d",x).start()] + '/' + x)

    return df





def class_id_to_label(id):

    label_map = {1: 'glass', 2: 'paper', 3: 'cardboard', 4: 'plastic', 5: 'metal', 6: 'trash'}

    return label_map[id]

    
IMAGES_DIR = '/kaggle/input/garbage-classification/Garbage classification/Garbage classification/'

    

train_file = '/kaggle/input/garbage-classification/one-indexed-files-notrash_train.txt'

val_file   = '/kaggle/input/garbage-classification/one-indexed-files-notrash_val.txt'

test_file  = '/kaggle/input/garbage-classification/one-indexed-files-notrash_test.txt'



df_train = pd.read_csv(train_file, sep=' ', header=None, names=['rel_path', 'label'])

df_valid = pd.read_csv(val_file,   sep=' ', header=None, names=['rel_path', 'label'])

df_test  = pd.read_csv(val_file,   sep=' ', header=None, names=['rel_path', 'label'])



df_train = add_class_name_prefix(df_train, 'rel_path')

df_valid = add_class_name_prefix(df_valid, 'rel_path')

df_test  = add_class_name_prefix(df_test,  'rel_path')



df_train['label'] = df_train['label'].apply(class_id_to_label)

df_valid['label'] = df_valid['label'].apply(class_id_to_label)

df_test['label']  = df_test['label'].apply(class_id_to_label)



print(f'Found {len(df_train)} training, {len(df_valid)} validation and {len(df_test)} samples.')
df_test.head()
datagen = ImageDataGenerator()



datagen_train = datagen.flow_from_dataframe(

    dataframe=df_train,

    directory=IMAGES_DIR,

    x_col='rel_path',

    y_col='label',

    color_mode="rgb",

    class_mode="categorical",

    batch_size=32,

    shuffle=True,

    seed=7,

)



datagen_valid = datagen.flow_from_dataframe(

    dataframe=df_valid,

    directory=IMAGES_DIR,

    x_col='rel_path',

    y_col='label',

    color_mode="rgb",

    class_mode="categorical",

    batch_size=32,

    shuffle=True,

    seed=7,

)
for i, img_path in enumerate(df_train.rel_path.sample(n=6, random_state=7)):

    img = load_img(IMAGES_DIR+img_path)

    img = img_to_array(img, dtype=np.uint8)

    

    #plt.figure(figsize = (5,5))

    plt.subplot(2, 3, i+1)

    plt.imshow(img.squeeze())
def build_model(num_classes):

    base_model = ResNet50(weights='imagenet', include_top=False)



    x = base_model.output

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(1024, activation='relu')(x)

    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)



    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)



    for layer in base_model.layers:

        layer.trainable = False

        

    return model





net = build_model(num_classes=6)



net.compile(optimizer='Adam',

            loss='categorical_crossentropy',

            metrics=[tf.keras.metrics.categorical_accuracy])



net.summary()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)



history = net.fit_generator(

    generator=datagen_train,

    validation_data=datagen_valid,

    epochs=30,

    validation_freq=1,

    callbacks=[early_stop]

)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 4))



axs[0].plot(history.history['loss'], label='loss')

axs[0].plot(history.history['val_loss'], label='val_loss')



axs[1].plot(history.history['categorical_accuracy'], label='acc')

axs[1].plot(history.history['val_categorical_accuracy'], label='val_acc')



plt.legend();

plt.show();
net.save('CNN-Modelo.h5')
test_generator = datagen.flow_from_dataframe(

    dataframe=df_test,

    directory=IMAGES_DIR,

    x_col='rel_path',

    y_col='label',

    color_mode="rgb",

    class_mode="categorical",

    batch_size=1,

    shuffle=False,

    seed=7

)



# y_pred = net.predict(test_generator, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)



filenames = test_generator.filenames

nb_samples = len(filenames)



metricas = net.evaluate_generator(test_generator, nb_samples)

metricas
#Cargamos nuevamente el modelo realizado

from keras.models import load_model

Salida = load_model('CNN-Modelo.h5')
#Carga de etiquetas del set de test

labels = (test_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())



print(labels)
#Generacion de 16 numeros random de imagenes de prueba

pruebasTest=np.random.randint(0,len(test_generator),16)

pruebasTest
#Mostrar predicciones y clasificacion real

plt.figure(figsize=(16, 16))

for i, img in enumerate(pruebasTest):

    a,b = test_generator.__getitem__(pruebasTest[i])

    PrediceImagen = Salida.predict(a)

    img = array_to_img(a[0])

    plt.subplot(4, 4, i+1)

    plt.title('pred: %s / truth: %s' % (labels[np.argmax(PrediceImagen)], labels[np.argmax(b)]))

    plt.imshow(img)
from keras.preprocessing import image
layer_outputs = [layer.output for layer in Salida.layers[:100]]

test_image = '../input/cnnclasificacionbasura/descarga.jpg'

activation_model = tf.keras.Model(inputs=Salida.input, outputs=layer_outputs)
img = image.load_img(test_image, target_size=(299, 299))

img_arr = image.img_to_array(img)

img_arr = np.expand_dims(img_arr, axis=0)

img_arr /= 255.
activations = activation_model.predict(img_arr)
plt.figure(figsize = (10,10))



plt.subplot(2,2,1),plt.imshow(activations[1][0, :, :, 1], cmap='plasma')

plt.title('Activacion 1'), plt.xticks([]), plt.yticks([])



plt.subplot(2,2,2),plt.imshow(activations[3][0, :, :, 1], cmap='plasma')

plt.title('Activacion 3'), plt.xticks([]), plt.yticks([])



plt.subplot(2,2,3),plt.imshow(activations[40][0, :, :, 1], cmap='plasma')

plt.title('Activacion 40'), plt.xticks([]), plt.yticks([])



plt.subplot(2,2,4),plt.imshow(activations[99][0, :, :, 1], cmap='plasma')

plt.title('Activacion 99'), plt.xticks([]), plt.yticks([])





plt.show()