import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

np.random.seed(0)
input_folder = '/kaggle/input'
# lendo input
df_train = pd.read_csv(input_folder+'/labels.csv')
df_test = pd.read_csv(input_folder+'/sample_submission.csv')
df_train.breed.value_counts().plot(kind='bar', figsize=(15,15), title="Quantidade de imagens por raça no treino");
df_train.head()
df_test.head()
targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse = True)
one_hot_labels = np.asarray(one_hot)
im_size = 224
from tqdm import tqdm # bliblioteca para colocar a porcentagem de andamento do for
import cv2 # biblioteca para visão computacional
x_train = []
y_train = []
x_test = []
i = 0 
for f, breed in tqdm(df_train.values):
    img = cv2.imread(input_folder+'/train/{}.jpg'.format(f))
    x_train.append(cv2.resize(img, (im_size, im_size)))
    label = one_hot_labels[i]
    y_train.append(label)
    i += 1
del df_train # apagando uma variável pra diminuir consumo de memória
for f in tqdm(df_test['id'].values):
    img = cv2.imread(input_folder+'/test/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (im_size, im_size)))
from sklearn.model_selection import train_test_split # biblioteca para fazer a divisão dos dados em treino e teste
num_class = 120
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, shuffle=True,  test_size=0.2, random_state=1)
from keras.preprocessing.image import ImageDataGenerator # biblioteca para data augmetantaion
datagen = ImageDataGenerator(width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.2,
                            rotation_range=30,
                            vertical_flip=False,
                            horizontal_flip=True) # aqui eu defino os parâmetros que irei 
                                                  # utilizar para gerar as imagens

train_generator = datagen.flow(np.array(X_train), np.array(Y_train), 
                               batch_size=32) 
valid_generator = datagen.flow(np.array(X_valid), np.array(Y_valid), 
                               batch_size=32) 
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Dropout, Flatten
from keras import regularizers
from keras.models import Model
base_model = InceptionV3(weights="imagenet",include_top=False, input_shape=(im_size, im_size, 3))
dropout = base_model.output
dropout = Dropout(0.5)(dropout)
model_with_dropout = Model(inputs=base_model.input, outputs=dropout)
    
x = model_with_dropout.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax',
                    kernel_regularizer=regularizers.l2(0.0015),
                    activity_regularizer=regularizers.l1(0.0015))(x)
    
my_model = Model(inputs=model_with_dropout.input, outputs=predictions)
        
my_model.compile(optimizer='sgd',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
my_model.fit_generator(
    train_generator,
    epochs=10, steps_per_epoch=len(X_train) / 18,
    validation_data=valid_generator, validation_steps=len(X_valid) / 18) # reali
preds = my_model.predict(np.array(x_test), verbose=1)
sub = pd.DataFrame(preds)
col_names = one_hot.columns.values
sub.columns = col_names
sub.insert(0, 'id', df_test['id'])
sub.head(5)
sub.to_csv("submission.csv")