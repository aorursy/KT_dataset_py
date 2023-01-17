# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import cv2

import os

import matplotlib.pyplot as plt

from keras import layers

from keras.models import Model,Sequential

from keras.layers import GlobalAveragePooling2D,Dense,Dropout

from keras.applications.vgg16 import VGG16,preprocess_input

from IPython.display import SVG

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler

from keras.models import model_from_json

print(os.listdir("../input/vgg16/"))

#print(os.listdir("../input/stanford-dogs-dataset/images/Images"))

#print(os.listdir("../input/annotations/Annotation"))





# Any results you write to the current directory are saved as output.
count=0

no_of_classes=10

folder_path="../input/stanford-dogs-dataset/images/Images"

folder=os.listdir("../input/stanford-dogs-dataset/images/Images")   

print(len(folder))

for i in range(no_of_classes):

        inner_folder=os.listdir(folder_path+"/"+folder[i])

        name=folder[i][10:] 

        print(name)

        for j in range(len(inner_folder)):

            img=cv2.imread(folder_path+"/"+folder[i]+"/"+inner_folder[j])

            count+=1

print(count)
def preprocessing():

    folder_path="../input/stanford-dogs-dataset/images/Images"

    folder=os.listdir("../input/stanford-dogs-dataset/images/Images")

    count=0

    no_of_classes=10

    X=np.zeros((1715,224,224,3))

    Y=np.zeros((1715,no_of_classes))

   

    labels=[]

    for i in range(no_of_classes):

        labels.append(folder[i][10:])



    print(labels)

    print(len(labels))

    for i in range(no_of_classes):

        inner_folder=os.listdir(folder_path+"/"+folder[i])

        name=folder[i][10:]

        print(name)

        for j in range(len(inner_folder)):

            tar=np.zeros(no_of_classes)

            img=cv2.imread(folder_path+"/"+folder[i]+"/"+inner_folder[j])

            img=cv2.resize(img,(224,224))    

            tar[labels.index(name)]=1

            #print(tar)

            X[count]=img/255.0

            Y[count]=tar

            count+=1

            #print(count,j)

        plt.imshow(X[count-1])

        plt.figure()

        print(Y[count-1])

    print(X.shape,Y.shape)

    return X,Y

X,Y=preprocessing()
def createModel():

    no_of_classes=10

    input_shapeDef=(224,224,3)

    basic_model=VGG16(include_top=False,input_shape=input_shapeDef,

                      weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    

    for layer in basic_model.layers:

        layer.trainable=False

    

    for layer in basic_model.layers:

        print(layer,layer.trainable)

        

    model=Sequential()

    model.add(basic_model)

    model.add(GlobalAveragePooling2D())

    model.add(Dropout(0.5))

    model.add(Dense(no_of_classes,activation='softmax'))



#     model=Model(basic_Model,out)

    

    model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

    

    return model
checkpoint = ModelCheckpoint(

    './base.model',

    monitor='val_loss',

    verbose=1,

    save_best_only=True,

    mode='min',

    save_weights_only=False,

    period=1

)

earlystop = EarlyStopping(

    monitor='val_loss',

    min_delta=0.001,

    patience=30,

    verbose=1,

    mode='auto'

)





reduce = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.1,

    patience=3,

    verbose=1, 

    mode='auto'

)



callbacks = [checkpoint,reduce]


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=69)

augs_gen = ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False, 

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        zca_whitening=False,  

        rotation_range=10,  

        zoom_range = 0.1, 

        width_shift_range=0.2,  

        height_shift_range=0.2, 

        horizontal_flip=True,  

        vertical_flip=False)



augs_gen.fit(X_train)


model=createModel()

model.summary()



h=model.fit_generator(augs_gen.flow(X_train,Y_train,batch_size=16),

                      steps_per_epoch=1000,

                      epochs=10,validation_data=[X_test,Y_test])

#save the model

model_json=model.to_json()

with open("model.json","w") as json_file:

    json_file.write(model_json)

    

#serialize weigths to HDF5

model.save_weights("model.h5")

print("Saved model to disk")
# load json and create model

json_file = open('model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")

print("Loaded model from disk")



 

# evaluate loaded model on test data

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

score = loaded_model.evaluate(X, Y, verbose=0)

print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
def predict(url):

    no_of_classes=10

    labels=[]

    folder=os.listdir("../input/stanford-dogs-dataset/images/Images")

    for i in range(no_of_classes):

        labels.append(folder[i][10:])



    print(labels)

    

    img=cv2.imread(url)

    if img is not None:

        img=cv2.resize(img,(224,224))

        img=img/255.0

        result=model.predict(img[np.newaxis,:,:,:])

        res=np.reshape(result,10)

        

        return labels[np.argmax(res)]

        

    else:

        return None

predict("../input/stanford-dogs-dataset/images/Images/n02091467-Norwegian_elkhound/n02091467_1110.jpg")
