#Import important libraries
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from tqdm import tqdm_notebook
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate,add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add ,Flatten ,Dense
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
# Read csv data files
train_data = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
test_data = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
train_data.shape #(60,000*785)
test_data.shape #(10000,785)
train_X= np.array(train_data.iloc[:,1:])
test_X= np.array(test_data.iloc[:,1:])
train_Y= np.array (train_data.iloc[:,0]) # (60000,)
test_Y = np.array(test_data.iloc[:,0]) #(10000,)
train_data.head()

train_X.shape, test_X.shape
# Convert the images into 3 channels to fit in input for transfer models
train_X=np.dstack([train_X] * 3)
test_X=np.dstack([test_X]*3)
train_X.shape,test_X.shape
# Reshape images as per the tensor format required by tensorflow
train_X = train_X.reshape(-1, 28,28,3)
test_X= test_X.reshape (-1,28,28,3)
train_X.shape,test_X.shape

# Resize the images 48*48 as required by transfer learning 
from keras.preprocessing.image import img_to_array, array_to_img
train_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in train_X])
test_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in test_X])
train_X.shape, test_X.shape
#specify Labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.title(class_names[train_Y[i]])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])    
    plt.imshow(train_X[i])
plt.show()
# Normalize the data and change data type
train_X = train_X / 255.
test_X = test_X / 255.

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

print(train_Y_one_hot)
print(test_Y_one_hot)
train_X,valid_X,train_label,valid_label = train_test_split(train_X,
                                                           train_Y_one_hot,
                                                           test_size=0.05,
                                                           random_state=13
                                                           )

imageSize=train_X[0].shape[0]
Channels=3
print("imageSize : ",imageSize)
# DenseNet169
from tensorflow.keras.applications  import DenseNet169 
preTrainedModelDenseNet169 = DenseNet169(input_shape = (imageSize, imageSize, Channels), 
                                include_top = False, 
                                weights=None)

preTrainedModelDenseNet169.load_weights("../input/densenet-keras/DenseNet-BC-169-32-no-top.h5")
DenseNet169layers=preTrainedModelDenseNet169.layers
print("Number of layer DenseNet169 : ",len(DenseNet169layers))
for layer  in range(len(DenseNet169layers)-250):
    DenseNet169layers[layer].trainable = False 
   


preTrainedModelDenseNet169.summary()
# ResNet152 V2 
from tensorflow.keras.applications   import ResNet152V2
preTrainedModelResNet152V2  = ResNet152V2 (input_shape = (imageSize, imageSize, Channels), 
                                include_top = False, 
                               weights="imagenet")


ResNet152V2layers=preTrainedModelResNet152V2.layers
print("Number of layer ResNet152V2 : ",len(ResNet152V2layers))
for layer  in range(len(ResNet152V2layers)-64): 
    ResNet152V2layers[layer].trainable = False 
    
    
preTrainedModelResNet152V2.summary()
# VGG-16
from tensorflow.keras.applications.vgg16  import VGG16
preTrainedModelVgg16 = VGG16(input_shape = (imageSize, imageSize, Channels), 
                                include_top = False, 
                                weights=None)
preTrainedModelVgg16.load_weights("../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
VGG16layers=preTrainedModelVgg16.layers
print("Number of layer Vgg16 : ",len(VGG16layers))
for layer  in range(len(VGG16layers)-5): 
       VGG16layers[layer].trainable = False 
        
preTrainedModelVgg16.summary()
#DenseNet169 Model
x=Flatten()(preTrainedModelResNet50.output)

#Fully Connection Layers

# FC1
x=Dense(1024, activation="relu")(x)

#Dropout to avoid overfitting effect
x=Dropout(0.4)(x)

# FC2
x=Dense(1024, activation="relu")(x)

# FC3
x=Dense(1024, activation="relu")(x)

#Dropout to avoid overfitting effect
x=Dropout(0.2)(x)

# FC4
x=Dense(512, activation="relu")(x)

# FC5
x=Dense(512, activation="relu")(x)

#Dropout to avoid overfitting effect
x=Dropout(0.2)(x)

# FC6
x=Dense(256, activation="relu")(x)

# FC7
x=Dense(256, activation="relu")(x)

#Dropout to avoid overfitting effect
x=Dropout(0.2)(x)

# FC8
x=Dense(128, activation="relu")(x)

#output layer
x=Dense(10,activation="softmax")(x)
#concatenation layers
modelDenseNet169=Model(preTrainedModelDenseNet169.input,x)
modelDenseNet169.summary()
#ResNet152V2 Model

x=Flatten()(preTrainedModelResNet152V2.output)

#Fully Connection Layers

# FC1
x=Dense(1024, activation="relu")(x)

# FC2
x=Dense(1024, activation="relu")(x)

# FC3
x=Dense(1024, activation="relu")(x)

# FC4
x=Dense(1024, activation="relu")(x)

# #Dropout to avoid overfitting effect
x=Dropout(0.2)(x)

# FC5
x=Dense(512, activation="relu")(x)

# FC6
x=Dense(512, activation="relu")(x)


# FC7
x=Dense(256, activation="relu")(x)

# FC8
x=Dense(256, activation="relu")(x)

# #Dropout to avoid overfitting effect
x=Dropout(0.2)(x)

#output layer
x=Dense(10,activation="softmax")(x)

#concatenation layers
modelResNet152V22=Model(preTrainedModelResNet152V2.input,x)
modelResNet152V2.summary()

#Vgg16 Model
x=Flatten()(preTrainedModelVgg16.output)

#Fully Connection Layer

# FC1
x=Dense(1024, activation="relu")(x)

# FC2
x=Dense(1024, activation="relu")(x)

# FC3
x=Dense(1024, activation="relu")(x)

#Dropout to avoid overfitting effect
x=Dropout(0.5)(x)

# FC4
x=Dense(512, activation="relu")(x)

# FC5
x=Dense(512, activation="relu")(x)

#Dropout to avoid overfitting effect
x=Dropout(0.4)(x)

# FC6
x=Dense(256, activation="relu")(x)

# FC7
x=Dense(64, activation="relu")(x)

# FC8
x=Dense(64, activation="relu")(x)

#Dropout to avoid overfitting effect
x=Dropout(0.2)(x)

#output layer
x=Dense(10,activation="softmax")(x)
#concatenation layers
modelVgg16=Model(preTrainedModelVgg16.input,x)
modelVgg16.summary()
#RMSPorp Optimization
optRMSProp=tf.keras.optimizers.RMSprop(
    learning_rate=0.0001,
    momentum=0.0001,
    epsilon=1e-07,
    name="RMSprop",
)

#compile DenseNet169  Model
modelDenseNet169.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['accuracy'])
#compile ResNet152V2 Model
modelResNet152V2.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['accuracy'])
#compile Vgg16 Model
modelVgg16.compile(optimizer=optRMSProp, loss="categorical_crossentropy",metrics=['accuracy'])
#Hyperparameters 
Epochs=100
BatchSize=700
#fit MobileNetV2 Model
historyResNet152V2Model=modelResNet152V2.fit(train_X,train_label,validation_data=(valid_X,valid_label) ,epochs=Epochs 
                                                     , batch_size =BatchSize ,verbose=1)  
#fit Vgg16 Model
historyVgg16Model=modelVgg16.fit(train_X,train_label,validation_data=(valid_X,valid_label) ,epochs=Epochs 
                                                    , batch_size =BatchSize   ,verbose=1)

#fit DenseNet169 Model

historyDenseNet169Model=modelDenseNet169.fit(train_X,train_label,validation_data=(valid_X,valid_label) ,epochs=Epochs 
                                                                , batch_size =BatchSize,verbose=1)



#Save ResNet50 Model
modelDanseNet196.save("WeightsForDanseNet196.h5")
print("Done for DenseNet169")
#Save ResNet152V2 Model
modelResNet152V2.save("WeightsForResNet152V2.h5")
print("Done for ResNet152V2")
#Save Vgg16 Model
modelVgg16.save("WeightsForVgg16.h5")
print("Done for Vgg16")

#DenseNet169 Model

print("- the Accuracy and Loss for DenseNet169 Model With 100 Epochs")
plt.figure(figsize=(40,20))
# summarize history for accuracy
plt.subplot(5,5,1)
plt.plot(historyDenseNet169Model.history['accuracy'])
plt.plot(historyDenseNet169Model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')


# summarize history for loss
plt.subplot(5,5,2)
plt.plot(historyDenseNet169Model.history['loss'])
plt.plot(historyDenseNet169Model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()
#ResNet152V2 Model

print("- the Accuracy and Loss for ResNet152V2 Model With 100 Epochs")

plt.figure(figsize=(40,20))
# summarize history for accuracy
plt.subplot(5,5,1)
plt.plot(historyResNet152V2Model.history['accuracy'])
plt.plot(historyResNet152V2Model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')


# summarize history for loss
plt.subplot(5,5,2)
plt.plot(historyResNet152V2Model.history['loss'])
plt.plot(historyResNet152V2Model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()


#Vgg16 Model

print("- the Accuracy and Loss for Vgg16 Model With 100 Epochs")

plt.figure(figsize=(40,20))
# summarize history for accuracy
plt.subplot(5,5,1)
plt.plot(historyVgg16Model.history['accuracy'])
plt.plot(historyVgg16Model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')


# summarize history for loss
plt.subplot(5,5,2)
plt.plot(historyVgg16Model.history['loss'])
plt.plot(historyVgg16Model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()
#Evaluate DenseNet169 Model
print("Evaluate DenseNet169  Model")
modelDenseNet169.evaluate(test_X,test_Y_one_hot)
#Evaluate ResNet152V2 Model
print("Evaluate ResNet152V2 Model")
modelResNet152V2.evaluate(test_X,test_Y_one_hot)
#Evaluate Vgg16 Model
print("Evaluate Vgg16 Model")
modelVgg16.evaluate(test_X,test_Y_one_hot)
historyMobileNetV2Model
#predict DenseNet169 Model
PredmodelDenseNet169=modelDenseNet169.predict(test_X)

#predict ResNet152V2 Model
PredmodelResNet152V2=modelResNet152V2.predict(test_X)

#predict Vgg16 Model
PredmodelVgg16= modelVgg16.predict(test_X)
#Process on Prediction values for DenseNet169 Model
PredmodelDenseNet169= np.argmax(PredmodelDenseNet169,axis=1)
print("Prediction values for DenseNet169 Model :\n",PredmodelDenseNet169)

#Process on Prediction values for MobileNetV2 Model
PredmodelResNet152V2= np.argmax(PredmodelResNet152V2,axis=1)
print("\nPrediction values for ResNet152V2 Model :\n",PredmodelResNet152V2)

#Process on Prediction values for Vgg16 Model
PredmodelVgg16= np.argmax(PredmodelVgg16,axis=1)
print("\nPrediction values for Vgg16 Model :\n",PredmodelVgg16)

#DenseNet169 model
plt.figure(figsize=(30,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.title(f"Predication: {class_names[PredmodelResNet50[i]]} <==> Truth: {class_names[test_Y[i]]}")
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])    
    plt.imshow(test_X[i])
plt.show()
#ResNet152V2 model
plt.figure(figsize=(30,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.title(f"Predication: {class_names[PredmodelResNet152V2[i]]} <==> Truth: {class_names[test_Y[i]]}")
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])    
    plt.imshow(test_X[i])
plt.show()
#Vgg16 model
plt.figure(figsize=(30,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.title(f"Predication: {class_names[PredmodelVgg16[i]]} <==> Truth: {class_names[test_Y[i]]}")
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])    
    plt.imshow(test_X[i])
plt.show()
#confusion matrix  for DenseNet169 Model
from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(test_Y,PredmodelDenseNet169)
print(cm)
#Visualizing confusion matrix
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                colorbar=True)
plt.show()
#confusion matrix  for ResNet152V2 Model
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(test_Y,PredmodelResNet152V2)
print(cm)
#Visualizing confusion matrix
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                colorbar=True)
plt.show()
#confusion matrix  for Vgg16 Model
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(test_Y,PredmodelVgg16)
print(cm)
#Visualizing confusion matrix
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                colorbar=True)
plt.show()
#Classification Report for DenseNet169 Model
from sklearn.metrics import classification_report
ClassificationReport = classification_report(test_Y,PredmodelDenseNet169)
print('Classification Report for DenseNet169 Model is : \n ', ClassificationReport )
#Classification Report for ResNet152V2 Model
from sklearn.metrics import classification_report
ClassificationReport = classification_report(test_Y,PredmodelResNet152V2)
print('Classification Report for ResNet152V2 Model is : \n ', ClassificationReport )
#Classification Report for Vgg16 Model
from sklearn.metrics import classification_report
ClassificationReport = classification_report(test_Y,PredmodelVgg16)
print('Classification Report for Vgg16 Model is : \n ', ClassificationReport )
