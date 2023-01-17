# Пути к файлам

labels_csv='../input/dog-breed-identification/labels.csv'

sample_submission_csv='../input/dog-breed-identification/sample_submission.csv'



# Пути до картинок

jpg_train='../input/dog-breed-identification/train/{}.jpg'  

jpg_test='../input/dog-breed-identification/test/{}.jpg'



# Настройки нейросети

im_resize = 120 # Размер изображения

num_class = 120 # Кол-во классов

batch_size = 32

Epochs = 64
import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt



import tensorflow.keras as keras

from keras import regularizers

from keras.models import Model

from keras.models import Sequential

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy

import os

from tqdm import tqdm

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import ImageDataGenerator
def gen_graph(history, title):

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('crossentropy ' + title)

    plt.ylabel('crossentropy')

    plt.xlabel('Epoch')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()

    

    plt.plot(history.history['categorical_accuracy'])

    plt.plot(history.history['val_categorical_accuracy'])

    plt.title('categorical_accuracy ' + title)

    plt.ylabel('categorical_accuracy')

    plt.xlabel('Epoch')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()
df_train = pd.read_csv(labels_csv) 

df_test = pd.read_csv(sample_submission_csv)
df_train.head()
df_test.head()
labels = df_train['breed']

one_hot = pd.get_dummies(labels, sparse = True)

one_hot_labels = np.asarray(one_hot)

#one_hot_labels
x_train = []

y_train = []

x_test = []
i = 0 

for f, breed in tqdm(df_train.values):

    img = load_img(jpg_train.format(f), target_size=(im_resize, im_resize))

    img_resized = img_to_array(img)

    x_train.append(img_resized)

    label = one_hot_labels[i]

    y_train.append(label)

    i += 1
for f in tqdm(df_test['id'].values):

    img = load_img(jpg_test.format(f), target_size=(im_resize, im_resize))

    img_resized = img_to_array(img)

    x_test.append(img_resized)
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, shuffle=True,  test_size=0.2)
del x_train, y_train, df_train
train_datagen = ImageDataGenerator(rotation_range=35, #поворот

                            rescale=1./255, #нормализация

                            horizontal_flip=True,

                            vertical_flip=True,

                            shear_range=15)



test_datagen = ImageDataGenerator(rescale=1./255)



# Создаем генераторы 

train_generator =train_datagen.flow(np.array(X_train), np.array(Y_train), 

                               batch_size=batch_size)

test_generator =test_datagen.flow(np.array(X_valid), np.array(Y_valid),

                              batch_size=batch_size*5)
# Создаем последовательную модель

model = Sequential()



# сверточный слой

model.add(Conv2D(32, (3, 3), padding='same',input_shape=(im_resize, im_resize, 3), activation='relu'))

# сверточный слой

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

# первый слой подвыборки

model.add(MaxPooling2D(pool_size=(2, 2)))

# Слой регуляризации Dropout

model.add(Dropout(0.20))





# сверточный слой

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

# сверточный слой

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

# Второй слой подвыборки

model.add(MaxPooling2D(pool_size=(2, 2)))

# Слой регуляризации Dropout

model.add(Dropout(0.20))





# сверточный слой

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

# сверточный слой

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

# Третий слой подвыборки

model.add(MaxPooling2D(pool_size=(2, 2)))

# Слой регуляризации Dropout

model.add(Dropout(0.20))





# Слой преобразования данных из 2D представления в плоское

model.add(Flatten())

# Полносвязный слой для классификации

model.add(Dense(512, activation='relu'))

# Слой регуляризации Dropout

model.add(Dropout(0.30))







# Выходной полносвязный слой

model.add(Dense(num_class, activation='softmax'))
model.compile(optimizer='adam',

          loss='categorical_crossentropy', 

           metrics=[categorical_accuracy])
print(model.summary())
from keras.callbacks import ModelCheckpoint, EarlyStopping



earlystop=EarlyStopping(monitor='val_loss', min_delta=0, patience=5)



checkpoint_callback = ModelCheckpoint('model_best.hdf5',

                                      monitor='val_categorical_accuracy', 

                                      save_best_only=True,

                                      verbose=1)
history = model.fit(

    train_generator,

    callbacks=[earlystop, checkpoint_callback],

    epochs=Epochs,

    steps_per_epoch=len(train_generator) // batch_size,

    

    validation_data=test_generator,

    validation_steps=len(test_generator) // batch_size)
#График точности на валидационной и обучающей выборке

gen_graph(history,

          "график точности")
from tensorflow.keras.models import load_model

model=load_model("model_best.hdf5")
preds = model.predict(np.array(x_test))
sub = pd.DataFrame(preds)

col_names = one_hot.columns.values

sub.columns = col_names

sub.insert(0, 'id', df_test['id'])

sub.head(5)
sub.to_csv("output_rmsprop_aug.csv", index=False)