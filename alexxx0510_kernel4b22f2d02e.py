import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import ast
import matplotlib.pyplot
import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.optimizers import SGD, adam
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from time import time
from keras.utils import to_categorical


#se carga el conjunto de datos 

initData = pd.read_csv('../input/finaldataset-29042020csv/finalDataset_29042020.csv')

data = initData
data.head()

#Se define un método para obtener las curvas BP y RP de cada registro
def getImages(band, simple):   
    p = band
    name='test.png'
    fig = plt.figure()
    if(simple==True):
      plt.plot(range(1,61),p, color='black')
    else:
      plt.plot(range(1,121),p, color='black')
    plt.axis('off')
    plt.show()
    plt.clf()
    fig.savefig('./'+name)   # save the figure to file
    
    plt.close(fig)

    image = Image.open('./'+name).convert('L')
    image.show()
    width, height = image.size  

    image = image.resize((48,72))
    image.show()
    return numpy.asarray(image)
data.head()
BPimages = []
RPimages = []
joinImages = []

for row in data.iterrows():
    bp = ast.literal_eval(row[1]['BP'])
    rp = ast.literal_eval(row[1]['RP'])
    joinImages.append(getImages(np.concatenate((bp,rp)),False))
    BPimages.append(getImages(bp,True))
    RPimages.append(getImages(rp,True))

data.head()
data['PlotJoin48x72']=joinImages
#data['PlotBP48x72']=BPimages
#data['PlotRP48x72']=RPimages
data
data.to_csv('dataPixel48x72.csv')
data = data.replace({'Class' : { 'SN Ia' : 0, 'ULENS' : 1, 'FLARE' : 2 }})
data

data[['PlotJoin48x72','Class']]
pixeles = []
for i in data.iterrows():
  aux = []
  for j in i[1].PlotJoin48x72:
    for f in j:
      aux.append(f)
  pixeles.append(aux)

pd.DataFrame(pixeles)
x=pd.concat([pd.DataFrame(pixeles), data['Class']],axis=1)
x.to_csv('dataPixel48x72.csv')
featuresJOIN = np.asarray(joinImages)
featuresBP = np.asarray(BPimages)
featuresRP = np.asarray(RPimages)
labels = data[['Class']].values.flatten()
print(featuresJOIN.shape)
print(featuresBP.shape)
print(featuresRP.shape)
featuresJOIN
train_x_orig_join, test_x_orig_join, train_y_orig_join, test_y_orig_join = train_test_split(featuresJOIN, labels, test_size=0.2, random_state=44)
train_x_orig_join.shape
train_x_orig_join.shape
train_x_join= train_x_orig_join.reshape(1871,72,48,1)/255
test_x_join= test_x_orig_join.reshape(468,72,48,1)/255
train_y_join = to_categorical(train_y_orig_join,3)
test_y_join = to_categorical(test_y_orig_join,3)
train_x_orig_bp, test_x_orig_bp, train_y_orig_bp, test_y_orig_bp = train_test_split(featuresBP, labels, test_size=0.2, random_state=44)
train_x_bp= train_x_orig_bp.reshape(1871,72,48,1)/255
test_x_bp= test_x_orig_bp.reshape(468,72,48,1)/255
train_y_bp = to_categorical(train_y_orig_bp,3)
test_y_bp = to_categorical(test_y_orig_bp,3)
train_x_orig_rp, test_x_orig_rp, train_y_orig_rp, test_y_orig_rp = train_test_split(featuresRP, labels, test_size=0.2, random_state=44)
train_x_rp= train_x_orig_rp.reshape(1871,72,48,1)/255
test_x_rp= test_x_orig_rp.reshape(468,72,48,1)/255
train_y_rp = to_categorical(train_y_orig_rp,3)
test_y_rp = to_categorical(test_y_orig_rp,3)
#MODELO SIMPLE
def createModelSimple(var):
  model = Sequential()

  if var == 1:
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu', strides=(1,1),
                    input_shape=(72,48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax"))
  return model

# MODELO VGG
def createVGGModel():
  modelVGG = Sequential()

  #Primera parte
  modelVGG.add(Conv2D(32, kernel_size=(3, 3), strides=(1,1),input_shape=(72,48, 1)))
  modelVGG.add(Activation("relu"))
  modelVGG.add(BatchNormalization(axis=-1))
  modelVGG.add(Conv2D(32, kernel_size=(3, 3), strides=(1,1)))
  modelVGG.add(Activation("relu"))
  modelVGG.add(BatchNormalization(axis=-1))
  modelVGG.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
  modelVGG.add(Dropout(0.25))


  #Segunda parte
              
  modelVGG.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1)))
  modelVGG.add(Activation("relu"))
  modelVGG.add(BatchNormalization(axis=-1))
  modelVGG.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1)))
  modelVGG.add(Activation("relu"))
  modelVGG.add(BatchNormalization(axis=-1))
  modelVGG.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
  modelVGG.add(Dropout(0.25))

  #Tercera parte
  modelVGG.add(Flatten())
  modelVGG.add(Dense(512))
  modelVGG.add(Activation("relu"))
  modelVGG.add(BatchNormalization(axis=-1))
  modelVGG.add(Dropout(0.5))
              
  #Cuarta parte
  modelVGG.add(Dense(3, activation="softmax"))

  return modelVGG

# MODELO VGG
def createVGG16Model():
  model = Sequential()
  model.add(Conv2D(input_shape=(72,48,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
  model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
  model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
  model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
  model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
  model.add(Flatten())
  model.add(Dense(units=4096,activation="relu"))
  model.add(Dense(units=4096,activation="relu"))
  model.add(Dense(units=3, activation="softmax"))


  return model
def lenetModel(): 
  model = keras.Sequential()

  model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(72,48, 1)))
  model.add(AveragePooling2D())

  model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
  model.add(AveragePooling2D())

  model.add(Flatten())

  model.add(Dense(units=120, activation='relu'))

  model.add(Dense(units=84, activation='relu'))

  model.add(Dense(units=3, activation = 'softmax'))
  return model
from keras import backend
backend.set_image_data_format('channels_last')
def alexnet(): 
  model = Sequential()
  model.add(Conv2D(96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(72,48,1)))
  model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

  model.add(Conv2D(256, kernel_size=(5,5), strides=(1,1), activation='relu'))
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

  model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu'))

  model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu'))

  model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

  model.add(Flatten())
  model.add(Dense(4096, input_shape=(72*48*1,), activation='relu'))
  model.add(Dropout(0.4))

  #model.add(Dense(4096, activation='relu'))
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.4))

  model.add(Dense(3, activation='softmax'))
  return model

from keras.applications import vgg16,vgg19,resnet,resnet_v2,inception_v3,inception_resnet_v2,mobilenet,mobilenet_v2,densenet,nasnet

model = keras.Sequential()

model.add(Conv2D(filters=12, kernel_size=(3, 3), activation='relu', input_shape=(72,48, 1)))
model.add(AveragePooling2D())

model.add(Conv2D(filters=16, kernel_size=(5, 5),activation='relu'))
model.add(AveragePooling2D())


model.add(Flatten())

model.add(Dense(units=120, activation='relu'))

model.add(Dense(units=84, activation='relu'))

model.add(Dense(units=3, activation = 'softmax'))
  
  
model.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])

fit = model.fit(train_x_join, train_y_join,
            validation_data=(test_x_join, test_y_join),
            batch_size=12,
            epochs=30)

#Evaluación el modelo sobre el conjunto de test
eva = model.evaluate(test_x_join, test_y_join, verbose=0)
print(eva[1])

from keras.applications import vgg16,vgg19,resnet,resnet_v2,mobilenet,mobilenet_v2,densenet,nasnet

modelCompiled=vgg16.VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(72,48, 1), pooling=None, classes=3)
modelCompiled.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.Adadelta(lr=0.01),
                    metrics=['accuracy'])

fit = modelCompiled.fit(train_x_join, train_y_join,
            validation_data=(test_x_join, test_y_join),
            batch_size=64,
            epochs=20)

#Evaluación el modelo sobre el conjunto de test
eva = modelCompiled.evaluate(test_x_join, test_y_join)
print(eva)
print(eva[1])




from tabulate import tabulate
from keras.applications import vgg16,vgg19,xception,resnet,resnet_v2,inception_v3,inception_resnet_v2,mobilenet,mobilenet_v2,densenet,nasnet

def executeMethods(trainX, trainY, testX, testY, models):
  results = []

  adadelta=[keras.optimizers.Adadelta(),'Adadelta']
  sgd=[keras.optimizers.Adadelta(),'SGD']
  adam=[keras.optimizers.Adam(),'Adam']
  adagrad=[keras.optimizers.Adagrad(),'Adagrad']
  adamax=[keras.optimizers.Adamax(),'Adamax']
  rsmprop=[keras.optimizers.RMSprop(),'RSMprop']

#Se quitan Nadam y batch_size=128 debido a que tras varias pruebas no convergen con ellos

  for batch in [12,32,64]:
    for opt in [sgd,adam,adadelta,rsmprop,adagrad,adamax]:


      for model in models:
        modelCompiled = 'x'

        if model == 'modelVGG':
          modelCompiled = createVGGModel()
        elif model == 'modelSimple':
          modelCompiled = createModelSimple(1)
        elif model == 'modelVGG16':
          modelCompiled = vgg16.VGG16(include_top=True, weights=None, input_shape=(72,48, 1), classes=3)
        elif model == 'modelVGG19':
          modelCompiled = vgg19.VGG19(include_top=True, weights=None, input_shape=(72,48, 1), classes=3)
        elif model == 'resnet50':
          modelCompiled = resnet.ResNet50(include_top=True, weights=None, input_shape=(72,48, 1), classes=3)
        elif model == 'resnet101':
          modelCompiled = resnet.ResNet101(include_top=True, weights=None, input_shape=(72,48, 1), classes=3)
        elif model == 'resnet152':
          modelCompiled = resnet.ResNet152(include_top=True, weights=None, input_shape=(72,48, 1), classes=3)
        elif model == 'resnet50v2':
          modelCompiled = resnet_v2.ResNet50V2(include_top=True, weights=None, input_shape=(72,48, 1), classes=3)
        elif model == 'resnet101v2':
          modelCompiled = resnet_v2.ResNet101V2(include_top=True, weights=None, input_shape=(72,48, 1), classes=3)
        elif model == 'resnet152v2':
          modelCompiled = resnet_v2.ResNet152V2(include_top=True, weights=None, input_shape=(72,48, 1), classes=3) 
        elif model == 'lenet':
          modelCompiled = lenetModel()
        elif model == 'mobileNet':
          modelCompiled = mobilenet.MobileNet(include_top=True, weights=None, input_shape=(72,48, 1), classes=3) 
          
        modelCompiled.compile(loss='categorical_crossentropy',
                    optimizer=opt[0],
                    metrics=['accuracy'])

        starttime=time()
        fit = modelCompiled.fit(trainX, trainY,
                  validation_data=(testX, testY),
                  batch_size=batch,
                  epochs=12)

        #Evaluación el modelo sobre el conjunto de test
        eva = modelCompiled.evaluate(testX, testY, verbose=0)
        results.append([model,eva[1],eva[0],batch,opt[1],time()-starttime])
      
  #Se ordena según la precisión
  results.sort(key = lambda x: x[1], reverse=True) 
  
  print(tabulate(results, headers=['Model','Accuracy', 'Loss','Batch Size','Optimizer','Tiempo de entrenamiento (sg)']))

models= ['lenet']
executeMethods(train_x_join,train_y_join,test_x_join,test_y_join,models)
models= ['modelSimple']
executeMethods(train_x_join,train_y_join,test_x_join,test_y_join,models)
models= ['modelVGG']
executeMethods(train_x_join,train_y_join,test_x_join,test_y_join,models)
models= ['resnet152']
executeMethods(train_x_join,train_y_join,test_x_join,test_y_join,models)