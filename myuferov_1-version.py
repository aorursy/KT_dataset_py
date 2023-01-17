im_resize = 64 # Размер изображения
num_class = 120 # Кол-во классов
Batch_Size = 256 # Размер минибатча 
Epochs=50 # Кол-во эпох
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt

import tensorflow.keras as keras
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.layers.convolutional import Conv2D, MaxPooling2D
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# для отрисовки результатов
def gen_graph(history, title):
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Accuracy ' + title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['fbeta'])
    plt.plot(history.history['fbeta'])
    plt.title('fbeta ' + title)
    plt.ylabel('MLogfbeta')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
from keras import backend
def fbeta(y_true, y_pred, beta=2):
    # clip predictions
    y_pred = backend.clip(y_pred, 0, 1)
    # calculate elements
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
    # calculate precision
    p = tp / (tp + fp + backend.epsilon())
    # calculate recall
    r = tp / (tp + fn + backend.epsilon())
    # calculate fbeta, averaged across each class
    bb = beta ** 2
    fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
    return fbeta_score


df_train = pd.read_csv('../input/dog-breed-identification/labels.csv') 
df_test = pd.read_csv('../input/dog-breed-identification/sample_submission.csv') 
jpg_train='../input/dog-breed-identification/train/{}.jpg'
jpg_test='../input/dog-breed-identification/test/{}.jpg'
df_train.head()
df_test.head()
labels = df_train['breed']
one_hot = pd.get_dummies(labels, sparse = True)
#one_hot
one_hot_labels = np.asarray(one_hot)
one_hot_labels
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
train_datagen = ImageDataGenerator(rotation_range=20,#поворот
                            rescale=1./255, #нормализация
                            horizontal_flip=True
                            )#разворот картинок
test_datagen = ImageDataGenerator(rescale=1./255) 
train_generator =train_datagen.flow(np.array(X_train), np.array(Y_train), 
                               batch_size=Batch_Size)
test_generator =test_datagen.flow(np.array(X_valid), np.array(Y_valid),
                              batch_size=Batch_Size)
#Создаем последовательную модель
model = Sequential()

# сверточный слой
model.add(Conv2D(32, (3, 3), padding='same',input_shape=(im_resize, im_resize, 3), activation='relu', kernel_initializer='he_uniform'))
# сверточный слой
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# Второй слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# сверточный слой
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# сверточный слой
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# Третий слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# сверточный слой
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# сверточный слой
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# Третий слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Слой преобразования данных из 2D представления в плоское
model.add(Flatten())
# Полносвязный слой для классификации
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.5))

# Выходной полносвязный слой
model.add(Dense(num_class, activation='sigmoid'))

model.compile(optimizer='adam',
          loss='binary_crossentropy', 
           metrics=[categorical_accuracy,fbeta])
print(model.summary())
from keras.callbacks import ModelCheckpoint, EarlyStopping

earlystop=EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=5)

checkpoint_callback = ModelCheckpoint('model_best.hdf5',
                                      monitor='val_categorical_accuracy', 
                                      save_best_only=True,
                                      verbose=1)
history = model.fit(
    train_generator,
    callbacks=[earlystop, checkpoint_callback],
    epochs=Epochs,
    steps_per_epoch=len(train_generator),  
    validation_data=test_generator,
    validation_steps=len(test_generator))
#График точности на валидационной и обучающей выборке
gen_graph(history,
          "график точности")
del history
from tensorflow.keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({"fbeta": fbeta})
model=load_model("model_best.hdf5")
preds = model.predict(np.array(x_test), verbose=0)
sub = pd.DataFrame(preds)
col_names = one_hot.columns.values
sub.columns = col_names
sub.insert(0, 'id', df_test['id'])
sub.head(5)
sub.to_csv("output_rmsprop_aug.csv", index=False)