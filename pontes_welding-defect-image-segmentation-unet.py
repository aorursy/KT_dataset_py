!pip install --upgrade tensorflow

!pip install segmentation-models

import numpy as np

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras.models import load_model

import keras

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate

from keras.models import Model

from keras.optimizers import *

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import segmentation_models as sm





import random

import os

import gc

import datetime

import numpy as np

import glob



from scipy.ndimage import rotate

from mpl_toolkits.axes_grid1 import ImageGrid

import PIL

from PIL import Image , ImageOps



import matplotlib.pyplot as plt



#tf.enable_eager_execution()     



# carrega o módulo do Tensorboard

# %load_ext tensorboard



print('A versão do tensorflow é: ' + tf.__version__)
DATA_DIR = '/kaggle/input/'

MODEL_PATH = 'models/'

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 256,256,1

AUGMENTATION_SAMPLE_SIZE=2000

ORIGINAL_DATASET_SIZE=1012
def load_data(d):

    x = np.load(d + 'x100.npy')

    t = np.load(d + 't100.npy')

    return x,t



def normalize(x):

    return x/255.



def print_images(n, x, t, y=None):

    if y:

        plt.subplot(131); plt.imshow(x[n,:,:,0], cmap='gray')

        plt.subplot(132); plt.imshow(t[n,:,:,0], cmap='gray')        

        plt.subplot(133); plt.imshow(y[n,:,:,0], cmap='gray')

    else:

        plt.subplot(121); plt.imshow(x[n,:,:,0], cmap='gray')

        plt.subplot(122); plt.imshow(t[n,:,:,0], cmap='gray')
def show_img(img, ax):

    ax.grid(False)

    ax.set_xticks([])

    ax.set_yticks([])

    ax.imshow(img,cmap='gray')



def plot_grid(imgs, nrows, ncols, figsize=(10, 10)):

    assert len(imgs) == nrows*ncols, f"Number of images should be {nrows}x{ncols}"

    _, axs = plt.subplots(nrows, ncols, figsize=figsize)

    axs = axs.flatten()

    for img, ax in zip(imgs, axs):

        show_img(img, ax)

            

def translate(img, shift=10, direction='right', roll=True):

    assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'

    img = img.copy()

    if direction == 'right':

        right_slice = img[:, -shift:].copy()

        img[:, shift:] = img[:, :-shift]

        if roll:

            img[:,:shift] = np.fliplr(right_slice)

    if direction == 'left':

        left_slice = img[:, :shift].copy()

        img[:, :-shift] = img[:, shift:]

        if roll:

            img[:, -shift:] = left_slice

    if direction == 'down':

        down_slice = img[-shift:, :].copy()

        img[shift:, :] = img[:-shift,:]

        if roll:

            img[:shift, :] = down_slice

    if direction == 'up':

        upper_slice = img[:shift, :].copy()

        img[:-shift, :] = img[shift:, :]

        if roll:

            img[-shift:,:] = upper_slice

    return img



def rotate_img(img, angle, bg_patch=(5,5)):

    img = img.copy()

    assert len(img.shape) <= 3, "Incorrect image shape"

    rgb = len(img.shape) == 3

    if rgb:

        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))

    else:

        #bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])

        bg_color = 0

    bg_color = 0

    img = rotate(img, angle, reshape=False)

    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]

    img[mask] = bg_color

    return img



def gaussian_noise(img, mean=0, sigma=0.03):

    img = img.copy()

    noise = np.random.normal(mean, sigma, img.shape)

    mask_overflow_upper = img+noise >= 1.0

    mask_overflow_lower = img+noise < 0

    noise[mask_overflow_upper] = 1.0

    noise[mask_overflow_lower] = 0

    img += noise

    return img



def generate_augmented(x, t, sample_size):

    newImages = np.ndarray(shape=(sample_size, x.shape[1], x.shape[2], x.shape[3]),

                     dtype=np.float32)

    newMasks = np.ndarray(shape=(sample_size, t.shape[1], t.shape[2], t.shape[3]),

                     dtype=np.float32) 

    

    for i in range(sample_size):

        idx = random.randint(0, len(x)-1)

    

        image = x[idx,:,:]

        mask = t[idx,:,:]

    

        newImg = image.copy()

        newMask = mask.copy()

    

        for orientation in ['up', 'down', 'left', 'right']:

            if random.random() > .8:

                shift = random.randint(10,128)

                newImg = translate(newImg, direction=orientation, shift=shift)

                newMask = translate(newMask, direction=orientation, shift=shift)

        

        if random.random() > .8:

            angle = random.randint(-180, 180)

            newImg = rotate_img(newImg, angle=angle)

            newMask = rotate_img(newMask, angle=angle)

            

        if random.random() > .5:

            newImg = gaussian_noise(newImg, 

                                    mean=random.uniform(0, 0.1), 

                                    sigma=random.uniform(0.00001, 0.0001)).reshape((256,256,1))

        newImages[i] = newImg

        newMasks[i] = newMask

        

    return newImages, newMasks
def dice_coef(y_true, y_pred, smooth=1.):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coef_loss(y_true, y_pred):

    return 1 - dice_coef(y_true, y_pred)
def save(model, name, path):

    # Save the model

    entire_path = path+name+'/'

    print('saving ', name)

    if not os.path.exists(entire_path):

        os.makedirs(entire_path)

    model.save(entire_path+name+'.h5')

    model.save_weights(entire_path+name+'_WEIGHTS.h5')

    print('saved successfully')



def load(name, path):

    # Recreate the exact same model purely from the file

    return load_model(path+name+'.h5')
#Carrega os dados e normaliza

x,t = load_data(DATA_DIR)



x = normalize(x)

t = normalize(t)
#Gera dados aplicando rotação, translação e geração de ruído

X_train_aug, y_train_aug = generate_augmented(x,t,AUGMENTATION_SAMPLE_SIZE)
#separa os dados reais entre validação e treinamento

X_train, X_validate, y_train, y_validate = train_test_split(x,t,test_size=0.33)

X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.5)



#concatena os dados de treinamento reais com os dados gerados

X_train = np.concatenate((X_train, X_train_aug))

y_train = np.concatenate((y_train, y_train_aug))



x = None

t = None



gc.collect()
#Modelo Unet

#...

def get_unet_model(optimizer, loss, metrics, pretrained_weights=None):

    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

 

    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                            padding='same')(inputs)

    c1 = tf.keras.layers.Dropout(0.1)(c1)

    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(c1)

    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)



    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(p1)

    c2 = tf.keras.layers.Dropout(0.1)(c2)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(c2)

    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)



    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(p2)

    c3 = tf.keras.layers.Dropout(0.2)(c3)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(c3)

    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)



    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(p3)

    c4 = tf.keras.layers.Dropout(0.2)(c4)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(c4)

    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)



    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(p4)

    c5 = tf.keras.layers.Dropout(0.3)(c5)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(c5)



    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)

    u6 = tf.keras.layers.concatenate([u6, c4])

    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(u6)

    c6 = tf.keras.layers.Dropout(0.2)(c6)

    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(c6)



    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)

    u7 = tf.keras.layers.concatenate([u7, c3])

    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(u7)

    c7 = tf.keras.layers.Dropout(0.2)(c7)

    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(c7)



    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)

    u8 = tf.keras.layers.concatenate([u8, c2])

    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(u8)

    c8 = tf.keras.layers.Dropout(0.1)(c8)

    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(c8)



    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)

    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)

    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(u9)

    c9 = tf.keras.layers.Dropout(0.1)(c9)

    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',

                                padding='same')(c9)



    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

 

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    

    if(pretrained_weights):

    	model.load_weights(pretrained_weights)

    

    return model
def treinar(model, x, y, nomedoexperimento, validation_split=0.15, validation_data=None, validation_steps=None, batch_size=16, patience=2, epochs=50, save_logs=False):   

    ### cria diretório "logs" com os arquivos para o callback do Tensorboard 

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    logdir = os.path.join('logs', nomedoexperimento + '-' + timestamp)



    checkpoint_path = "training_1/"+nomedoexperimento+".ckpt"

#     checkpoint_dir = os.path.dirname(checkpoint_path)

 

    # Create checkpoint callback

#     cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 

#                                                     save_weights_only=True,

#                                                     verbose=1)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)

    

    callbacks = [

        tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss'),

        

#         cp_callback 

        checkpointer

    ]

    

#     if(save_logs):

#         callbacks.append(tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0))



    

    if validation_data:

        results = model.fit(x, y, 

                            validation_data=validation_data, 

                            validation_steps=validation_steps,

                            batch_size=batch_size, 

                            epochs=epochs,

                            callbacks=callbacks)

    else:

        results = model.fit(x, y, 

                            validation_split=validation_split, 

                            batch_size=batch_size, 

                            epochs=epochs,

                            callbacks=callbacks)

    

    return results

 

 
model = get_unet_model(optimizer='adam', 

                       loss=dice_coef_loss, 

                       metrics=[dice_coef])



model.summary()



model = None

gc.collect()
size = 5

start = 30



for i in range(start, start+size):

    plot_grid([X_train_aug[i,:,:,0], y_train_aug[i,:,:,0]], 1,2, figsize=(5,5)),

    



X_train_aug = None

y_train_aug = None

gc.collect()
print('Tamanho do dataset inicial: ', ORIGINAL_DATASET_SIZE)

print('Tamanho do dataset gerado por augmentation: ', AUGMENTATION_SAMPLE_SIZE)

print('Tamanho do a serem utilizados para treinamento ', len(X_train))

print('Tamanho do a serem utilizados para validação ', len(X_validate))

print('Tamanho do a serem utilizados para teste/avaliação ', len(X_test))
#Otimizador Adam

#Loss dice

#Validação definida no split

#100 épocas e patience 5

experimento = "adam_dice_vdata_b50_p5"



model_adam_final = get_unet_model(optimizer='adam', 

                       loss=dice_coef_loss, 

                       metrics=[dice_coef])



opt_adam_history = treinar(model_adam_final, X_train, y_train,

        experimento, 

        validation_data=(X_validate, y_validate),

        batch_size=50, patience=5, epochs=80, save_logs=True)



save(model_adam_final, experimento, MODEL_PATH)
results = model_adam_final.evaluate(X_test, y_test, batch_size=50)

print(model_adam_final.metrics_names)

print('Loss dataset teste: ', results[0])

print('Dice dataset teste: ', results[1])
#Otimizador Nadam

#Loss dice

#Validação definida no split

#100 épocas e patience 5

experimento = "nadam_dice_vdata_b50_p5"



model = get_unet_model(optimizer='nadam', 

                       loss=dice_coef_loss, 

                       metrics=[dice_coef])



opt_nadam_history = treinar(model, X_train, y_train,

        experimento, 

        validation_data=(X_validate, y_validate),

        batch_size=50, patience=5, epochs=80)



save(model, experimento, MODEL_PATH)
#validação nadam

results = model.evaluate(X_test, y_test, batch_size=50)

print(model.metrics_names)

print('Loss dataset teste: ', results[0])

print('Dice dataset teste: ', results[1])
#Otimizador SGD

#Loss dice

#Validação definida no split

#100 épocas e patience 5

experimento = "sgd_dice_vdata_b50_p5"



model = get_unet_model(optimizer='sgd', 

                       loss=dice_coef_loss, 

                       metrics=[dice_coef])



opt_sgd_history = treinar(model, X_train, y_train,

        experimento, 

        validation_data=(X_validate, y_validate),

        batch_size=50, patience=5, epochs=80)



save(model, experimento, MODEL_PATH)
#validação sgd

results = model.evaluate(X_test, y_test, batch_size=50)

print(model.metrics_names)

print('Loss dataset teste: ', results[0])

print('Dice dataset teste: ', results[1])
# Plot training & validation loss values

plt.plot(opt_adam_history.history['loss'])

plt.plot(opt_nadam_history.history['loss'])

plt.plot(opt_sgd_history.history['loss'])

plt.title('Model train loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Adam Train', 'Nadam Train', 'SGD Train'], loc='upper left')

plt.show()



plt.plot(opt_adam_history.history['val_loss'])

plt.plot(opt_nadam_history.history['val_loss'])

plt.plot(opt_sgd_history.history['val_loss'])

plt.title('Model validation loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Adam val', 'Nadam val','SGD val'], loc='upper left')

plt.show()
del opt_adam_history

del opt_nadam_history

del opt_sgd_history



gc.collect()
#Otimizador Adam

#Loss dice

#Validação definida no split

#100 épocas e patience 5

experimento = "b20"



model = get_unet_model(optimizer='adam', 

                       loss=dice_coef_loss, 

                       metrics=[dice_coef])



b20_history = treinar(model, X_train, y_train,

        experimento, 

        validation_data=(X_validate, y_validate),

        batch_size=20, patience=5, epochs=80)



save(model, experimento, MODEL_PATH)
#validação sgd

results = model.evaluate(X_test, y_test, batch_size=20)

print(model.metrics_names)

print('Loss dataset teste: ', results[0])

print('Dice dataset teste: ', results[1])
#Otimizador Adam

#Loss dice

#Validação definida no split

#100 épocas e patience 5

experimento = "b60"



model = get_unet_model(optimizer='adam', 

                       loss=dice_coef_loss, 

                       metrics=[dice_coef])



b60_history  = treinar(model, X_train, y_train,

        experimento, 

        validation_data=(X_validate, y_validate),

        batch_size=60, patience=5, epochs=80)



save(model, experimento, MODEL_PATH)
#validação sgd

results = model.evaluate(X_test, y_test, batch_size=60)

print(model.metrics_names)

print('Loss dataset teste: ', results[0])

print('Dice dataset teste: ', results[1])
#Otimizador Adam

#Loss dice

#Validação definida no split

#100 épocas e patience 5

experimento = "b100"



model = get_unet_model(optimizer='adam', 

                       loss=dice_coef_loss, 

                       metrics=[dice_coef])



b100_history = treinar(model, X_train, y_train,

        experimento, 

        validation_data=(X_validate, y_validate),

        batch_size=100, patience=5, epochs=80)



save(model, experimento, MODEL_PATH)
#validação sgd

results = model.evaluate(X_test, y_test, batch_size=100)

print(model.metrics_names)

print('Loss dataset teste: ', results[0])

print('Dice dataset teste: ', results[1])
# Plot training & validation loss values

plt.plot(b20_history.history['loss'])

plt.plot(b60_history.history['loss'])

plt.plot(b100_history.history['loss'])

plt.title('Model train loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Batch 20', 'Batch 60', 'Batch 100'], loc='upper left')

plt.show()



plt.plot(b20_history.history['val_loss'])

plt.plot(b60_history.history['val_loss'])

plt.plot(b100_history.history['val_loss'])

plt.title('Model validation loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Batch 20', 'Batch 60','Batch 100'], loc='upper left')

plt.show()
del b20_history

del b60_history

del b100_history



gc.collect()
def get_keras_callbacks(nomedoexperimento, patience):

    directory = 'training_1/'

    if not os.path.exists(directory):

        os.makedirs(directory)

    

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    logdir = os.path.join('logs', nomedoexperimento + '-' + timestamp)

    checkpoint_path = "training_1/"+nomedoexperimento+".ckpt"

    checkpointer = keras.callbacks.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1)



    return [

        keras.callbacks.callbacks.EarlyStopping(patience=patience, monitor='val_loss'),

        checkpointer

    ]
#Modelo pré treinado

# keras.backend.set_image_data_format('channels_last')

sm.set_framework('keras')



N = X_train.shape[-1]



base_model = sm.Unet('resnet34', classes=1, activation='sigmoid', encoder_weights='imagenet')



inp = Input(shape=(None, None, N))

l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels

out = base_model(l1)



model = Model(inp, out, name=base_model.name)



model.compile(

    'Adam',

    loss=dice_coef_loss,

    metrics=[dice_coef],

)



# del base_model

# gc.collect()
experimento = "pre_trained_unet_resnet34_imagenet_adam_b50_p3_vdata"



pre_trained_unet_resnet_imagenet_history = model.fit(

    x=X_train,

    y=y_train,

    batch_size=50,

    epochs=50,

    validation_data=(X_validate, y_validate),

    callbacks=get_keras_callbacks(experimento, patience=5)

)



save(model, experimento, MODEL_PATH)
results = model.evaluate(X_test, y_test, batch_size=50)

print(model.metrics_names)

print('Loss dataset teste: ', results[0])

print('Dice dataset teste: ', results[1])
#Modelo pré treinado

# keras.backend.set_image_data_format('channels_last')

N = X_train.shape[-1]



base_model = sm.Linknet('inceptionv3', classes=1, activation='sigmoid', encoder_weights='imagenet')



inp = Input(shape=(None, None, N))

l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels

out = base_model(l1)



model = Model(inp, out, name=base_model.name)



model.compile(

    'Adam',

    loss=dice_coef_loss,

    metrics=[dice_coef],

)
experimento = "pre_trained_linknet_inceptionv3_imagenet_adam_b50_p5_vdata"



pre_trained_linknet_inceptionv3_imagenet_history = model.fit(

    x=X_train,

    y=y_train,

    batch_size=50,

    epochs=50,

    validation_data=(X_validate, y_validate),

    callbacks=get_keras_callbacks(experimento, patience=5)

)



save(model, experimento, MODEL_PATH)
results = model.evaluate(X_test, y_test, batch_size=50)

print(model.metrics_names)

print('Loss dataset teste: ', results[0])

print('Dice dataset teste: ', results[1])
# Plot training & validation loss values

plt.plot(pre_trained_unet_resnet_imagenet_history.history['loss'])

plt.plot(pre_trained_linknet_inceptionv3_imagenet_history.history['loss'])

plt.title('Model train loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Unet ResNet Imagenet', 'LinkNet InceptionV3 Imagenet'], loc='upper left')

plt.show()



plt.plot(pre_trained_unet_resnet_imagenet_history.history['val_loss'])

plt.plot(pre_trained_linknet_inceptionv3_imagenet_history.history['val_loss'])

plt.title('Model validation loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Unet ResNet Imagenet', 'LinkNet InceptionV3 Imagenet'], loc='upper left')

plt.show()
del pre_trained_unet_resnet_imagenet_history

del pre_trained_linknet_inceptionv3_imagenet_history

gc.collect()
#Otimizador Adam

#Loss dice

#Validação definida no split

#100 épocas e patience 5

# weights_file = "models/adam_dice_vdata_b50_p5/adam_dice_vdata_b50_p5_WEIGHTS.h5"



# model_adam_final = get_unet_model(optimizer='adam', 

#                        loss=dice_coef_loss, 

#                        metrics=[dice_coef])



# model_adam_final.load_weights(weights_file)





# saida = model_adam_final.predict(X_test)