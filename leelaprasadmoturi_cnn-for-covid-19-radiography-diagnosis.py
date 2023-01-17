import os
import numpy as np
import pandas as pd
from shutil import copyfile

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint  
from keras import applications
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
!mkdir model
!mkdir dataset
!mkdir ./dataset/train
!mkdir ./dataset/train/NORMAL
!mkdir ./dataset/train/COVID-19
!mkdir ./dataset/train/Viral\ Pneumonia
!mkdir ./dataset/validation
!mkdir ./dataset/validation/NORMAL
!mkdir ./dataset/validation/COVID-19
!mkdir ./dataset/validation/Viral\ Pneumonia
!mkdir ./dataset/test
!mkdir ./dataset/test/NORMAL
!mkdir ./dataset/test/COVID-19
!mkdir ./dataset/test/Viral\ Pneumonia
def copy_samples(X, y, files_path, dataset_path, samples_path ):
    train_df = pd.DataFrame(X, columns=['path'])
    train_df['label'] = y
    for index, row in train_df.iterrows():
        row_split = row['path'].split('/')
        file_name = row_split[-1]
        label = row_split[-2]
        copyfile(f'{files_path}/{label}/{file_name}', f'{dataset_path}/{samples_path}/{label}/{file_name}')
imagePaths = []
files_path = '/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/'
dataset_path = './dataset/'

for dirname, _, filenames in os.walk(files_path):
    for filename in filenames:
        if (filename[-3:] == 'png'):
            label = dirname.split('/')[-1]
            path = os.path.join(dirname, filename)
            imagePaths.append((path, label))

imagePaths[0]

df_paths = pd.DataFrame(imagePaths, columns=['path', 'label'])

(X_train, X_validation, y_train, y_validation) = train_test_split(df_paths['path'], df_paths['label'],test_size=0.2, stratify=df_paths['label'], random_state=42)
(X_validation, X_test, y_validation, y_test) = train_test_split(X_validation, y_validation,test_size=0.1, stratify=y_validation, random_state=42)

copy_samples(X_train, y_train, files_path, dataset_path, 'train')
copy_samples(X_validation, y_validation, files_path, dataset_path, 'validation')
copy_samples(X_test, y_test, files_path, dataset_path, 'test')
width = 128
height = 128
batch_size = 16

VGG16_model = applications.VGG16(include_top=False, weights='imagenet')
def pre_process(path):

    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
            path,
            target_size=(width, height),
            batch_size=batch_size,
            class_mode=None, 
            shuffle=False)

    data = VGG16_model.predict_generator(generator)
    count_covid19 = len([file_path for file_path in generator.filepaths if 'COVID-19' in file_path])
    count_normal = len([file_path for file_path in generator.filepaths if 'NORMAL' in file_path])
    count_pneumonia = len([file_path for file_path in generator.filepaths if 'Viral Pneumonia' in file_path])
    
    labels = np.array([0] * count_covid19 + [1] * count_normal + [2] * count_pneumonia)
    return data, to_categorical(labels)
train_data, train_labels = pre_process('./dataset/train')
validation_data, validation_labels = pre_process('./dataset/validation')
test_data, test_labels = pre_process('./dataset/test')
checkpoint_path = '/kaggle/working/model/weights.best.VGG16.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)

history = model.fit(train_data, train_labels,
          epochs=50,
          batch_size=batch_size,
          validation_data=(validation_data, validation_labels),
          callbacks=[checkpointer],
          verbose = 1)
import matplotlib.pyplot as plt
import numpy

plt.figure(figsize=(10,5)) 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure(figsize=(10,5)) 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
from sklearn.metrics import classification_report

model.load_weights(checkpoint_path)
predictions = model.predict(test_data, batch_size=16)
predictions = np.argmax(predictions, axis=1)
print(classification_report(test_labels.argmax(axis=1), predictions, target_names=['Covid-19', 'Normal', 'Viral Pneumonia'], digits = 3))
model.save('cnnmodel.h5')
VGG16_model.save('vgg.h5')
model.save('cnn.h5')