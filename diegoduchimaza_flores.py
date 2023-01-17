# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/flowers-recognition"))

# Any results you write to the current directory are saved as output.
data = "../input/flowers-recognition/flowers/flowers"
folders = os.listdir(data)
print(folders)

import cv2
from tqdm import tqdm
image_names = []
train_labels = []
train_images = []

size = 120,120

for folder in folders:
    for file in tqdm(os.listdir(os.path.join(data,folder))):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data,folder,file))
            train_labels.append(folder)
            img = cv2.imread(os.path.join(data,folder,file))
            im = cv2.resize(img,size)
            train_images.append(im)
        else:
            continue


X1 = np.array(train_images)

X1.shape
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

le=LabelEncoder()
Y=le.fit_transform(train_labels)
Y=to_categorical(Y,5)
X=np.array(X1)
X=X/255
print(X.shape)
print(Y.shape)
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.20, random_state=42)
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)

import matplotlib.pyplot as plt
import seaborn as sns
for i in range(10,100,20):
    plt.imshow(X_train[i][:,:,0],cmap='gray')
    plt.show()
del X,Y
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AveragePooling2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'valid', 
                 activation ='relu', input_shape = (120,120,3)))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters =128, kernel_size = (3,3),padding = 'valid', 
                 activation ='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.2)) 

model.add(Conv2D(filters = 160, kernel_size = (3,3),padding = 'valid',
                 activation ='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 192, kernel_size = (3,3),padding = 'valid',
                 activation ='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.2)) 

model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'valid',
                 activation ='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.2)) 

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(5, activation = "softmax"))
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 50
batch_size = 256

datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False, 
        rotation_range=10, 
        zoom_range = 0.8,
        width_shift_range=0.8,  
        height_shift_range=0.8,  
        horizontal_flip=False,  
        vertical_flip=False)  

datagen.fit(X_train) 

model.summary()
%%time
history = model.fit(X_train,Y_train, batch_size=batch_size, epochs = epochs, validation_data = (X_val,Y_val))
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.models import model_from_json


from keras import layers
from keras import models



def guardarRNN(model,nombreArchivoModelo,nombreArchivoPesos):
    print("Guardando Red Neuronal en Archivo")  
    # serializar modelo a JSON

    # Guardar los Pesos (weights)
    model.save_weights(nombreArchivoPesos+'.h5')

    # Guardar la Arquitectura del modelo
    with open(nombreArchivoModelo+'.json', 'w') as f:
        f.write(model.to_json())

    print("Red Neuronal Grabada en Archivo")   
    
def cargarRNN(nombreArchivoModelo,nombreArchivoPesos):
        
    # Cargar la Arquitectura desde el archivo JSON
    with open(nombreArchivoModelo+'.json', 'r') as f:
        model = model_from_json(f.read())

    # Cargar Pesos (weights) en el nuevo modelo
    model.load_weights(nombreArchivoPesos+'.h5')  

    print("Red Neuronal Cargada desde Archivo") 
    return model

model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=0)
model.summary()
print('Resultado en Train:')
score = model.evaluate(X_train, Y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#Fase de Testing
print('Resultado en Test:')
score = model.evaluate(X_val, Y_val, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

nombreArchivoModelo='arquitectura_base'
nombreArchivoPesos='pesos_base'
guardarRNN(model,nombreArchivoModelo,nombreArchivoPesos)
X=[]
Z=[]
IMG_SIZE=150
FLOWER_DAISY_DIR='../input/flowers-recognition/flowers/daisy'
FLOWER_SUNFLOWER_DIR='../input/flowers-recognition/flowers/sunflower'
FLOWER_TULIP_DIR='../input/flowers-recognition/flowers/tulip'
FLOWER_DANDI_DIR='../input/flowers-recognition/flowers/dandelion'
FLOWER_ROSE_DIR='../input/flowers-recognition/flowers/rose'
def assign_label(img,flower_type):
    return flower_type
#
def make_train_data(flower_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,flower_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        X.append(np.array(img))
        Z.append(str(label))
make_train_data('Daisy',FLOWER_DAISY_DIR)
print(len(X))
make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)
print(len(X))
make_train_data('Tulip',FLOWER_TULIP_DIR)
print(len(X))

make_train_data('Rose',FLOWER_ROSE_DIR)
print(len(X))
fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Flower: '+Z[l])
        
plt.tight_layout()
le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,5)
X=np.array(X)
X=X/255
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.models import model_from_json

from keras.applications.resnet import ResNet50
from keras import layers
from keras import models



def guardarRNN(model,nombreArchivoModelo,nombreArchivoPesos):
    print("Guardando Red Neuronal en Archivo")  
    # serializar modelo a JSON

    # Guardar los Pesos (weights)
    model.save_weights(nombreArchivoPesos+'.h5')

    # Guardar la Arquitectura del modelo
    with open(nombreArchivoModelo+'.json', 'w') as f:
        f.write(model.to_json())

    print("Red Neuronal Grabada en Archivo")   
    
def cargarRNN(nombreArchivoModelo,nombreArchivoPesos):
        
    # Cargar la Arquitectura desde el archivo JSON
    with open(nombreArchivoModelo+'.json', 'r') as f:
        model = model_from_json(f.read())

    # Cargar Pesos (weights) en el nuevo modelo
    model.load_weights(nombreArchivoPesos+'.h5')  

    print("Red Neuronal Cargada desde Archivo") 
    return model
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5, activation = "softmax"))

batch_size=128
epochs=50

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)
datagen = ImageDataGenerator(
        featurewise_center=False, # establece la media de entrada a 0 sobre el conjunto de datos
        samplewise_center=False,  # establece cada media de muestra en 0
        featurewise_std_normalization=False, # dividir entradas por estándar del conjunto de datos 
        samplewise_std_normalization=False, # divide cada entrada por su estándar
        zca_whitening=False,  # aplicar blanqueamiento ZCA
        rotation_range=10,  # rotar imágenes al azar en el rango (grados, 0 a 180)
        zoom_range = 0.1, # Ampliar aleatoriamente la imagen
        width_shift_range=0.2, # mover imágenes al azar horizontalmente (fracción del ancho total) 
        height_shift_range=0.2, # mover imágenes al azar verticalmente (fracción de la altura total)
        horizontal_flip=True, # voltear imágenes al azar
        vertical_flip=False) # voltear imágenes al azar 


datagen.fit(x_train)
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data = (x_test,y_test))



#Cargar pesos y la arquitectura
model2=cargarRNN(nombreArchivoModelo,nombreArchivoPesos) 

model2.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['acc']) #ADADELTA: An Adaptive Learning Rate Method
score = model2.evaluate(x_train, y_train, verbose=0)
print('Resultado en Train:')
print("%s: %.2f%%" % (model2.metrics_names[1], score[1]*100))

#Fase de Testing
print('Resultado en Test:')
score = model2.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model2.metrics_names[1], score[1]*100))

#Guardamos los archivos de modelo de pruebas
nombreArchivoModelo='arquitectura_prueba'
nombreArchivoPesos='pesos_prueba'
guardarRNN(model,nombreArchivoModelo,nombreArchivoPesos)
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()
plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()
# obteniendo predicciones sobre el conjunto de valores.
pred=model.predict(x_test)
pred_digits=np.argmax(pred,axis=1)
# ahora almacena algunos índices correctamente y mal clasificados '.
i=0
prop_class=[]
mis_class=[]

for i in range(len(y_test)):
    if(np.argmax(y_test[i])==pred_digits[i]):
        prop_class.append(i)
    if(len(prop_class)==8):
        break

i=0
for i in range(len(y_test)):
    if(not np.argmax(y_test[i])==pred_digits[i]):
        mis_class.append(i)
    if(len(mis_class)==8):
        break
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[prop_class[count]])
        ax[i,j].set_title("Predicted Flower :"+str(le.inverse_transform([pred_digits[prop_class[count]]]))+"\n"+"Actual Flower : "+str(le.inverse_transform([np.argmax(y_test[prop_class[count]])])))
        plt.tight_layout()
        count+=1
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[mis_class[count]])
        ax[i,j].set_title("Predicted Flower :"+str(le.inverse_transform([pred_digits[mis_class[count]]]))+"\n"+"Actual Flower : "+str(le.inverse_transform([np.argmax(y_test[mis_class[count]])])))
        plt.tight_layout()
        count+=1
%%time
history = model.fit(x_train,y_train, batch_size=batch_size, epochs = epochs, validation_data = (x_test,y_test))
#1. Compilación: Prueba de mejores parámetros batch_size, epochs y optimizer
#Esto recomiendo probarlo con Google Colab, puesto que se necesita 16GB en RAM y puede llegar a tardar unos 30min.

def build_model(optimizer):
    model = Sequential()
    model.add(Dense(32, input_shape=(x_train.shape[1],), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model


#parámetros que queremos probar, y sus valores 
#probaremos con batch_size, epochs, y optimizador, con el fin de encontrar la mejor combinación entre estos tres parámetros.
parameters = parameters = {'batch_size': [16,32],
             'epochs':[100,500],
             'optimizer': ['adadelta', 'rmsprop']}

estimator = KerasClassifier(build_fn=build_model, verbose=0)
#Ahora no le pasamos los parámetros al KerasClasifier, porque se los pasaremos a través de GridSearchCV
#el argumento verbose=0 es para que no muestre salida, si lo dejamos en cero, no mostrará la barra de progreso del entrenamiento
#GridSearchCV: recibe como parámetros nuestro modelo, nuestros parámetros, la medida sobre la que queremos comparar, y la 
#cantidad de veces que lo entrenará para sacar la media de accuracy.
grid_search = GridSearchCV(estimator=estimator, param_grid=parameters, scoring='accuracy', cv=10,n_jobs=-1)
grid_search.fit(x_train, y_train)
#grid_search.best_params_
print(grid_search.best_params_)
#Un ejemplo de resultados es: {'batch_size': 16, 'epochs': 100, 'optimizer': 'rmsprop'}
#Esto indica que el optimizador "adadelta" no es adecuado. Y es que este optimizador NO sirve para este tipo de problemas.

