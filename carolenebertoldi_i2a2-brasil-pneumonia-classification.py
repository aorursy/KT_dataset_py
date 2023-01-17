%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import pandas as pd
df_train = pd.read_csv('/kaggle/input/i2a2-brasil-pneumonia-classification/train.csv')
df_train.head()
positive = df_train.query('pneumonia == 1')
print(len(positive))

negative = df_train.query('pneumonia == 0')
print(len(negative))
def show_images(df):
    for filename in df['fileName']:
      image = mpimg.imread(f"/kaggle/input/i2a2-brasil-pneumonia-classification/images/{filename}")
      imgplot = plt.imshow(image)
      plt.show()
df = positive.head()
show_images(df)
df = negative.head()
show_images(df)
df_test = pd.read_csv('/kaggle/input/i2a2-brasil-pneumonia-classification/test.csv')
df_test.head()
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

from keras.callbacks import EarlyStopping, ModelCheckpoint   
from keras.layers import Conv2D, Dense, Dropout, GlobalMaxPooling2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
MODEL_PATH = 'model.pneumonia.weights.best.hdf5'
def init_model():
    if not os.path.isfile(MODEL_PATH):
        model = train()
        print('New train!')
    else:
        model = create_model()
        model.load_weights(MODEL_PATH)
        print('Using network trained!')

    return model
def prepare_dataset(df):
    images = [preprocess_images(f"/kaggle/input/i2a2-brasil-pneumonia-classification/images/{filename}") for filename in df['fileName']]
    images = np.array(images, dtype=np.float32)

    outputs = None
    if 'pneumonia' in df.columns:
        outputs = df['pneumonia']
    
    return images, outputs

def preprocess_images(filename):
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
    return preprocess_input(image) 
def create_model():
    base_model = InceptionResNetV2(weights="imagenet",
                          classes=2,
                          classifier_activation="softmax",
                          include_top=False, 
                          input_shape= (224, 224, 3))

    x = GlobalMaxPooling2D()(base_model.output)
    x = Dense(16, activation = 'relu')(x)
    x = Dense(1, activation = 'sigmoid')(x)

    model = Model(base_model.input, x)
    model.summary()
    
    return model
def train():
    df_train = pd.read_csv('/kaggle/input/i2a2-brasil-pneumonia-classification/train.csv')
    images, outputs = prepare_dataset(df_train)
    
    # divindo dataset de treinamento em treinamento, teste e validação
    x_train, x_test, y_train, y_test = train_test_split(images, outputs, test_size = 0.2, stratify = outputs)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, stratify = y_train)

    # normalização
    x_train = x_train.astype('float32')/255
    x_valid = x_valid.astype('float32')/255
    x_test = x_test.astype('float32')/255

    model = create_model()
    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.BinaryCrossentropy(), 
                  metrics=['accuracy'])
    
    checkpointer = [ModelCheckpoint(filepath=MODEL_PATH, save_best_only=True),
                    EarlyStopping(patience= 15)]
    
    aug = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")
    
    class_weight = {
        0: 1.37044467,
        1: 3.69945848
    }
        
    hist = model.fit(
       x=aug.flow(x_train, y_train, batch_size=64),
       epochs=50,
       validation_data=(x_valid, y_valid),
       class_weight = class_weight,
       callbacks=checkpointer)

    # carregando os pesos que geraram a melhor precisão de validação
    model.load_weights(MODEL_PATH)

    # avaliar e imprimir a precisão do teste
    score = model.evaluate(x_test, y_test)
    print('\n', 'Test accuracy:', score[1])
    
    return model
train()
def predict_value(value):
    if value > 0.5:
        return 1
    return 0
    
def execute_prediction(df):
    X, _ = prepare_dataset(df)

    X = X.astype('float32')/255
    model = init_model()

    predictions = model.predict(X)
    predictions = [predict_value(pred) for pred in predictions]
    
    filenames = df['fileName']
    return pd.DataFrame({ "fileName": filenames, "pneumonia": predictions })
df_test = pd.read_csv('/kaggle/input/i2a2-brasil-pneumonia-classification/test.csv')
result_test = execute_prediction(df_test)
result_test.head(10)
df_submission = pd.read_csv('/kaggle/input/i2a2-brasil-pneumonia-classification/sample_submission.csv')
result_submission = execute_prediction(df_submission)
result_submission.head(10)
result_submission.to_csv("results.csv",index=False)