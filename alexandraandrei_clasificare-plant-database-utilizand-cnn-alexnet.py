import os
os.listdir("../input/plant-diseases-classification-using-alexnet")
# Importare librarii
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 
import cv2 as cv

# Initializarea retelei neurale convolutionale
retea = Sequential()

# Convolution Step 1
retea.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))

# Max Pooling Step 1
retea.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
retea.add(BatchNormalization())

# Convolution Step 2
retea.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))

# Max Pooling Step 2
retea.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
retea.add(BatchNormalization())

# Convolution Step 3
retea.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
retea.add(BatchNormalization())

# Convolution Step 4
retea.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
retea.add(BatchNormalization())

# Convolution Step 5
retea.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))

# Max Pooling Step 3
retea.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
retea.add(BatchNormalization())

# Flattening Step
retea.add(Flatten())

# Full Connection Step
retea.add(Dense(units = 4096, activation = 'relu'))
retea.add(Dropout(0.4))
retea.add(BatchNormalization())
retea.add(Dense(units = 4096, activation = 'relu'))
retea.add(Dropout(0.4))
retea.add(BatchNormalization())
retea.add(Dense(units = 1000, activation = 'relu'))
retea.add(Dropout(0.2))
retea.add(BatchNormalization())
retea.add(Dense(units = 38, activation = 'softmax'))
retea.summary()
retea.load_weights('../input/plant-diseases-classification-using-alexnet/best_weights_9.hdf5')
# Vizualizarea stucturii retelei, a straturilor
from keras import layers
for i, layer in enumerate(retea.layers):
   print(i, layer.name)
#Se antreneaza reteaua folosint primele 2 blocuri convolutionale, se "blocheaza" primele opt staturi
print("Straturi:")
for i, layer in enumerate(retea.layers[:20]):
    print(i, layer.name)
    layer.trainable = False

#Vizualizare parametrii. parametrii antrenabili scad dupa inghetarea unor straturi de jos
retea.summary()
# Rularea retelei
from keras import optimizers
retea.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# procesarea imaginilor
from keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')
valid_data = ImageDataGenerator(rescale=1./255)

batch_size = 128
base_dir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"

training_set = train_data.flow_from_directory(base_dir+'/train',
                                                 target_size=(224, 224),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')

valid_set = valid_data.flow_from_directory(base_dir+'/valid',
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            class_mode='categorical')
print("Afisarea claselor si a indicilor:")
clase_plante = training_set.class_indices
print(clase_plante)
print("Afisarea claselor: ")
clase = list(clase_plante.keys())
print(clase)
train_num = training_set.samples
valid_num = valid_set.samples
# checkpoint
from keras.callbacks import ModelCheckpoint
weightpath = "best_weights_9.hdf5"
checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [checkpoint]

#clasificarea imaginilor
history = retea.fit_generator(training_set,
                         steps_per_epoch=train_num//batch_size,
                         validation_data=valid_set,
                         epochs=5,
                         validation_steps=valid_num//batch_size,
                         callbacks=callbacks_list)
#salvarea modelului
filepath="AlexNetModel.hdf5"
retea.save(filepath)
#afisarea parametrilor obtinuti in urma antrenarii
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

#accuracy plot
plt.plot(epochs, acc, color='green', label='Acuratetea de antrenare')
plt.plot(epochs, val_acc, color='blue', label='Acuratetea de testare')
plt.title('Reprezentarea evolutiei valorii acuratetetii pentru antrenare si testare')
plt.ylabel('Acuratete')
plt.xlabel('Epoca')
plt.legend()

plt.figure()
#loss plot
plt.plot(epochs, loss, color='purple', label='Pierderi antrenare')
plt.plot(epochs, val_loss, color='red', label='Pierderi testare')
plt.title('Reprezentarea evolutiei valorii pierderilor pentru antrenare si testare')
plt.xlabel('Epoca')
plt.ylabel('Pierdere')
plt.legend()

plt.show()

print("Valoarea acuratetii pentru fiecare epoca:")
print(val_acc)
# Exemplu de clasificare a unei imagini. Predictia clasei
from keras.preprocessing import image
import numpy as np
image_path = "../input/new-plant-diseases-dataset/test/test/AppleCedarRust1.JPG"
imag = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(imag)
img = np.expand_dims(img, axis=0)
img = img/255

print("Following is our prediction:")
prediction = retea.predict(img)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
d = prediction.flatten()
j = d.max()
for index,item in enumerate(d):
    if item == j:
        class_name = clase[index]

##Another way
# img_class = retea.predict_classes(img)
# img_prob = retea.predict_proba(img)
# print(img_class ,img_prob )


#ploting image with predicted class name        
plt.figure(figsize = (4,4))
plt.imshow(imag)
plt.axis('off')
plt.title(class_name)
plt.show()