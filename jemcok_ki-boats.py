# het eerste wat gedaan wordt is de libraries inladen
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from keras.preprocessing import image
# voor het model
from keras.applications import VGG16
from keras.models import Sequential, load_model, Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPool2D, GlobalMaxPooling2D

from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras import regularizers

seed=123
random.seed(seed)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf 

if tf.test.gpu_device_name(): 
    print('\n\nDefault GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")
# hier krijg je 2 lijsten met de locaties van de foto's en een locatie met de 5 categorie namen
files = glob.glob('../input/kunstmatigeintelligentie20192020/Archive/Category*/*.jpg') # creerd een lijst met alle locatie's van de foto's

path="../input/kunstmatigeintelligentie20192020/Archive"
names = (os.listdir(path))
names.sort()
print(names)
print(files[0])
# hier zijn 5 functies voor wat visualisatie

#kijken naar de dimensie van de verschillende foto's
# Dit is zelf al onderzorgd en er is in ons opzichte de beste verhouding gekozen width, height, channels
def dimensie():
    array = pd.DataFrame(columns=('width', 'height'))
    for fname in files:
        img = image.load_img(fname)
        x = image.img_to_array(img)
        width = len(x[0])
        height = len(x)
        df = pd.DataFrame([[width, height]], columns=('width', 'height'))
        array = array.append(df)
    
    width = int(round(np.mean(array['width']))+1)
    height = 139
    
    return (height+6), width, len(x[0,0]), array
    

# lijst met 5 samples van elke categorie voor visualisatie
def plotlijst():
    plot_lijst=[]
    aantal=[]
    for i in range(1, 6):
        lenght = len(glob.glob('../input/kunstmatigeintelligentie20192020/Archive/Category%s/*.jpg' % i))
        aantal.append(lenght)
        files = random.sample(glob.glob('../input/kunstmatigeintelligentie20192020/Archive/Category%s/*.jpg' % i), 5)
        plot_lijst.append(files)    
    
    return plot_lijst, aantal



# functie voor visualisatie
def visueel(plot_lijst, height, width):        
    fig = plt.figure(figsize=(17, 30))
    plt.subplots_adjust(left=.2, bottom=.5)
    for row, lijst in enumerate(plot_lijst):        
        for col, index in enumerate(lijst):
            img = image.load_img(index, target_size=(height, width))   
            subplot = fig.add_subplot(5, len(lijst), row*len(lijst)+col+1)
            subplot.set_title('Category %s' % (row+1))
            for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
                    label.set_fontsize(10)
            plt.imshow(img)

def acc_loss_plot(history):
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc)+1)
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    
    plt.show()
    
    
def layer_outputs(model, img):    
    
    img_tensor = image.img_to_array(img)
    img_tensor = img_tensor.reshape((1,) + img_tensor.shape)
    #length = len(model.layers)
    layer_outputs = [layer.output for layer in model.layers[:4]]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    
    activations = activation_model.predict(img_tensor)    
    
    layer_names = []
    for layer in model.layers[:4]:
        layer_names.append(layer.name)
    
    images_per_row = 8
    
    # Now let's display our feature maps
    for layer_name, layer_activation in zip(layer_names, activations):
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]
    
        # The feature map has shape (1, size, size, n_features)
        height = layer_activation.shape[1]
        width = layer_activation.shape[2]
        
    
        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((height * n_cols, images_per_row * width))
    
        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * height : (col + 1) * height,
                             row * width : (row + 1) * width] = channel_image
    
        # Display the grid
        scale = 1. / height
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        
    plt.show()  
plot_lijst, aantal = plotlijst()
sns.barplot(x=names, y=aantal) # plot van de verdeling

max(aantal) / sum(aantal) # als er alleen gekozen wordt op category 1 heb je een acc van 34%

height=139
width=210
channel=3
visueel(plot_lijst, height, width)


probability = [i / sum(aantal) for i in aantal]
random_base = sum([(tal*prob)/sum(aantal) for tal, prob in zip(aantal, probability)])
print(round(random_base*100,2))
batch_size=32
train_datagen = image.ImageDataGenerator(
      rescale=1/255,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest',
      validation_split=.2)

train_generator = train_datagen.flow_from_directory(
        path,
        target_size=(height, width),
        color_mode='rgb',
        class_mode='categorical',
        classes=names,
        batch_size=batch_size,
        subset='training'
        )

val_generator = train_datagen.flow_from_directory(
        path,
        target_size=(height, width),
        color_mode='rgb',
        class_mode='categorical',
        classes=names,
        batch_size=batch_size,
        subset='validation'
        )
img = image.load_img('../input/kunstmatigeintelligentie20192020/Archive/Category1/2780099.jpg')

x = image.img_to_array(img)
print(x.shape)
x = x.reshape((1,)+x.shape)

i = 0
for batch in train_datagen.flow(x, batch_size=1):
    
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
    
plt.show()
vgg16 =load_model('../input/modellen1/vgg16.h5')
"""
vgg16 = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(height, width, channel),
             pooling='max')
"""
vgg16.summary()
model = Sequential()
model.add(vgg16)
model.add(Dense(4096, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=1e-6),
        metrics=['acc'])

num_epochs = 60

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(.1), patience=3, min_lr=2e-7, verbose=1)
callbacks = [EarlyStopping(patience=5), reduce_lr] #ModelCheckpoint('model_eind.h5', save_best_only=True),

history  = model.fit_generator(train_generator,
                               steps_per_epoch=int(5002*.8/batch_size),
                               epochs=num_epochs,
                               validation_data=val_generator,
                               validation_steps=int(5002*.2/batch_size),
                               callbacks=callbacks)
model.evaluate_generator(val_generator)
acc_loss_plot(history)
model = load_model('../input/modellen1/best_model.h5')
test_files=glob.glob('../input/kunstmatigeintelligentie20192020/Images/Images/*.jpg')
test_images= [image.load_img(x, target_size=(height, width)) for x in test_files]
test_images = np.array([image.img_to_array(x) for x in test_images])

test_images = test_images.astype('float32') / 255

lijst = os.listdir('../input/kunstmatigeintelligentie20192020/Images/Images')

predict = model.predict_classes(test_images)+1

  
data = {'id':lijst,
        'Category':predict}

df = pd.DataFrame(data, columns = ['id', 'Category'])
df.head()
model1 = load_model('../input/modellen/max_adam_4096_lr7_3.h5')
model2 = load_model('../input/modellen/model_best_acc_1.h5')
model3 = load_model('../input/modellen/model_best_acc_4.h5')
model4 = load_model('../input/modellen1/best_model.h5')
model5 = load_model('../input/modellen/dense_rmsprop_best.h5')


test_files=glob.glob('../input/kunstmatigeintelligentie20192020/Images/Images/*.jpg')  # data locaties
# laat afbeeldingen in ipv flow_from_directory
test_images= [image.load_img(x, target_size=(height, width)) for x in test_files] 
# Maakt er arrays van om te kunnen voorspelen
test_images = np.array([image.img_to_array(x) for x in test_images])
# rescale
test_images = test_images.astype('float32') / 255

model_list = [model1, model2, model3, model4, model5]
y = np.empty((1250), dtype=int)

# maakt lijst met voorspelingen
for model in model_list:
    predict = model.predict_classes(test_images)+1
    
    y = np.column_stack((y, predict))

# gebruikt de meest voorkomende voorspelling
y = y[:,1:]
voorspel = np.array([], dtype=int)
for i in y:
    count = np.bincount(i)
    voorspel = np.append(voorspel, np.argmax(count))
    
    


lijst = os.listdir('../input/kunstmatigeintelligentie20192020/Images/Images')
    
data = {'id':lijst,
        'Category':voorspel}
    
df = pd.DataFrame(data, columns = ['id', 'Category'])

print(y[0:5,])
print(df.head())
regu_rate = 1e-5

model = Sequential()
 
# block
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(height, width, channel),
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same',
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(MaxPool2D((2,2), strides=(2,2)))
 
# block
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(MaxPool2D((2,2), strides=(2,2)))
 
# block
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2()))
model.add(MaxPool2D((2,2), strides=(2,2)))
 
# block
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(MaxPool2D((2,2), strides=(2,2)))
 
# block
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l2(regu_rate)))
model.add(MaxPool2D((2,2), strides=(2,2)))
 
# block #
model.add(GlobalMaxPooling2D())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
 


model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

print(model.summary())