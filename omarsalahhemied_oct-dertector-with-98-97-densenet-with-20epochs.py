# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session by
import matplotlib.pyplot as plt 
import cv2 as cv
image = plt.imread("../input/kermany2018/OCT2017 /train/DME/DME-1072015-1.jpeg")
plt.imshow(image,cmap='gray')
image=cv.resize(image,(300,300))
image=image.reshape(300,300,1)
image.shape
import os 
TrainPath="../input/kermany2018/OCT2017 /train/"
TestPath="../input/kermany2018/OCT2017 /test/"
ValidationPath="../input/kermany2018/OCT2017 /val/"

print(TrainPath)
print(TestPath)
print(ValidationPath)
trainClass=os.listdir(TrainPath)
for i in range(len(trainClass)):
    imageInfile=[]
    imageInfile=os.listdir(os.path.join(TrainPath,trainClass[i]))
    print( f"the numbers of images in {trainClass[i]} class : " , len(imageInfile))
    plt.figure(figsize=(20,20))
    for j in range(5):
        
        plt.subplot(1,5,j+1)
        image=plt.imread(os.path.join(os.path.join(TrainPath,trainClass[i]),imageInfile[j]))
        plt.title(trainClass[i])
        plt.imshow(image,cmap='gray')
    print(image.shape)
    plt.show()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
imageDelegate=ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    
    )

ImageSize=224
trainGenerator=imageDelegate.flow_from_directory(
    TrainPath,
    batch_size=100,
    target_size=(ImageSize,ImageSize)  
)

testGenerator =imageDelegate.flow_from_directory(
     TestPath,
     batch_size= 10,
     target_size=(ImageSize, ImageSize)
)

validationGenerator =imageDelegate.flow_from_directory(
     ValidationPath,
     batch_size=4,
     target_size=(ImageSize, ImageSize)
)
Labels={0:"BME",1:"CNV",2:"DRUSEN",3:"NORMAL"}
def GetLabel(key):
    return Labels[key]


print(GetLabel(0))
print(GetLabel(1))
print(GetLabel(2))
print(GetLabel(3))
plt.figure(figsize=(20,20))
for i  in range(20):
    plt.subplot(4,5,i+1)
    plt.imshow(trainGenerator.__getitem__(0)[0][i])
    plt.title(GetLabel(np.argmax(trainGenerator.__getitem__(0)[1][i])))
    
    

trainGenerator.__getitem__(0)[0][1].shape
# map_characters = {0: 'Normal', 1: 'CNV', 2: 'DME', 3: 'DRUSEN'}
# dict_characters=map_characters
# import seaborn as sns
# df = pd.DataFrame()
# df["labels"]=y_train
# lab = df['labels']
# dist = lab.value_counts()
# sns.countplot(lab)
# print(dict_characters)
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Flatten ,Dropout , Dense  ,Conv2D , MaxPooling2D ,BatchNormalization 
# model=Sequential()
# model.add(Conv2D(50,(3,3),input_shape =(300,300,3),activation="relu"))
# model.add(MaxPooling2D((3,3)))

# model.add(Flatten())
# model.add(BatchNormalization())
# model.add(Dense(256, activation="relu"))
# model.add(Dense(4,activation="sigmoid"))

# #Optimzation
# model.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])

# epochs=10

# #Fit model
# history=model.fit_generator(testGenerator,epochs=epochs,steps_per_epoch=96
#                                     ,validation_data=validationGenerator ,validation_steps=32/4,verbose=1 )
from tensorflow.keras.applications.densenet  import DenseNet169
preTrainedModelDenseNet169 = DenseNet169(input_shape =( ImageSize, ImageSize, 3), include_top = False, 
weights = None
                                        )
preTrainedModelDenseNet169.load_weights("../input/densenet-keras/DenseNet-BC-169-32-no-top.h5")
for layer in preTrainedModelDenseNet169.layers:
    layer.trainable = False  
preTrainedModelDenseNet169.summary()

from tensorflow.keras import Model
#DenseNet169 Model
x=Flatten()(preTrainedModelDenseNet169.output)

#Fully Connection Layers
# FC1
x=Dense(1024, activation="relu")(x)
x=BatchNormalization()(x)
x=Dense(512, activation="relu")(x)
#Dropout to avoid overfitting effect
x=Dropout(0.2)(x)
# FC2
x=Dense(256, activation="relu")(x)
x=Dense(128, activation="relu")(x)


#output layer
x=Dense(4,activation="sigmoid")(x)


modelDenesNet=Model(preTrainedModelDenseNet169.input,x)
modelDenesNet.summary()
#Optimzation
modelDenesNet.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])

epochs=20

#Fit model  """83484/50=1670 steps_per_epoch=the number of image /batch size"""
history=modelDenesNet.fit_generator(trainGenerator,epochs=epochs,steps_per_epoch=835,validation_data=validationGenerator ,validation_steps=8,verbose=1 )
modelDenesNet.save("modelDenesNet.h5")
modelDenesNet.evaluate(testGenerator)
print("- the Accuracy and Loss for DenesNet169 Model With 20 Epochs")
plt.figure(figsize=(40,20))
# summarize history for accuracy
plt.subplot(5,5,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')



# summarize history for loss
plt.subplot(5,5,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()
from tensorflow.keras.applications.vgg19  import VGG19
preTrainedModelVGG19 = VGG19(input_shape =( ImageSize, ImageSize, 3), include_top = False)
# preTrainedModelVGG19.load_weights("../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")
for layer in preTrainedModelVGG19.layers:
    layer.trainable = False  
preTrainedModelVGG19.summary()
from tensorflow.keras import Model

#mobileNet Model
x=Flatten()(preTrainedModelVGG19.output)

#Fully Connection Layers
# FC1
x=Dense(1024, activation="relu")(x)
x=Dense(512, activation="relu")(x)
#Dropout to avoid overfitting effect
x=Dropout(0.2)(x)
# FC2
x=Dense(256, activation="relu")(x)
x=Dense(128, activation="relu")(x)


#output layer
x=Dense(4,activation="sigmoid")(x)


ModelVGG19=Model(preTrainedModelVGG19.input,x)
ModelVGG19.summary()

#Optimzation
ModelVGG19.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])
epochs=20
#Fit model
historyModelVGG19=ModelVGG19.fit_generator(trainGenerator,epochs=epochs,steps_per_epoch=835,validation_data=validationGenerator ,validation_steps=8,verbose=1 )
ModelVGG19.save("ModelVGG19.h5")
ModelVGG19.evaluate(testGenerator)
import matplotlib.pyplot as plt 
print("- the Accuracy and Loss for vgg19 Model With 20 Epochs")
plt.figure(figsize=(40,20))
# summarize history for accuracy
plt.subplot(5,5,1)
plt.plot(historyModelVGG19.history['accuracy'])
plt.plot(historyModelVGG19.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')



# summarize history for loss
plt.subplot(5,5,2)
plt.plot(historyModelVGG19.history['loss'])
plt.plot(historyModelVGG19.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()
from tensorflow.keras.applications.mobilenet  import MobileNet
preTrainedModelMobileNet = MobileNet(input_shape =( ImageSize, ImageSize, 3), include_top = False)
for layer in preTrainedModelMobileNet.layers:
    layer.trainable = False  
preTrainedModelMobileNet.summary()
from tensorflow.keras import Model

#MobileNet Model
x=Flatten()(preTrainedModelMobileNet.output)

#Fully Connection Layers
# FC1
x=Dense(1024, activation="relu")(x)
x=Dense(512, activation="relu")(x)
#Dropout to avoid overfitting effect
x=Dropout(0.2)(x)

x=Dense(256, activation="relu")(x)

# FC2
x=Dense(128, activation="relu")(x)


#output layer
x=Dense(4,activation="sigmoid")(x)


ModelMobileNet=Model(preTrainedModelMobileNet.input,x)
ModelMobileNet.summary()

#Optimzation
ModelMobileNet.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])

#Fit model
historyModelMobileNet=ModelMobileNet.fit_generator(trainGenerator,epochs=epochs,steps_per_epoch=835 ,validation_data=validationGenerator,validation_steps=8 ,verbose=1 )
ModelMobileNet.save("ModelMobileNet.h5")
ModelMobileNet.evaluate(testGenerator)
import matplotlib.pyplot as plt 
print("- the Accuracy and Loss for MobileNet Model With 10 Epochs")
plt.figure(figsize=(40,20))
# summarize history for accuracy
plt.subplot(5,5,1)
plt.plot(historyModelMobileNet.history['accuracy'])
plt.plot(historyModelMobileNet.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')



# summarize history for loss
plt.subplot(5,5,2)
plt.plot(historyModelMobileNet.history['loss'])
plt.plot(historyModelMobileNet.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()
