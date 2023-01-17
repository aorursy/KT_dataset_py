import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import cv2
import os
from random import choice, shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from zipfile import ZipFile


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
print(tf.__version__)
#[TODO] Extrair o dataset
train_data = ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip')
test_data = ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip')

train_data.extractall()
test_data.extractall()
TRAIN_DIR = os.path.abspath('./train')
TEST_DIR = os.path.abspath('./test')
IMG_SIZE = 128
#0 DOG/ 1 CAT
def label_image(img):
    word_label = img.split('.')[-3]
    if word_label == 'dog': return 1
    elif word_label == 'cat': return 0

def create_train_data():
    train_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_image(img)
        path = os.path.join(TRAIN_DIR,img)
        
        #ABRE A IMAGEM EM ESCALA DE CINZA -> cv2.imread(path,cv2.IMREAD_GRAYSACLE)
        #REDIMENSIONA A IMAGEM PARA (50 50) -> cv.resive(img,(x,y))
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR),(IMG_SIZE,IMG_SIZE))
        train_data.append([np.array(img),np.array(label)])
        
    shuffle(train_data)
    return train_data
  

def process_test_data():
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR),(IMG_SIZE,IMG_SIZE))
        test_data.append([np.array(img).astype(np.float32),img_num])
    return test_data

data_train = create_train_data()
data_test = process_test_data() 
TRAIN = data_train[250:]
VALID = data_train[:250]

x_predict = np.array([img[0] for img in data_test])
number_image = np.array([img[1] for img in data_test])

x_train = np.array([i[0] for i in TRAIN])
y_train = np.array([i[1] for i in TRAIN])

x_test = np.array([i[0] for i in VALID])
y_test = np.array([i[1] for i in VALID])
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3),weights='imagenet', include_top=False)
base_model.trainable = True
base_model.summary()
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output_layer = tf.keras.layers.Dense(units=1,activation='sigmoid')(global_average_layer)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.000001),loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(x_train,y_train,epochs=10,batch_size=64)
model.evaluate(x_test,y_test)
predictions = model.predict(x_predict)
predict = pd.Series(predictions.squeeze())
predict = pd.Series(np.where(predict>0.5,1,0).squeeze())
result = {'id':number_image,'label':predict}
result = pd.DataFrame(data=result)
result.to_csv('result.csv',index=False)
result