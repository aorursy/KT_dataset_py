# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from skimage.transform import resize





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Metadata")

df.head(19)

#lets drop previous exist coloum (shows that image exists in the directory or not)

df=df.drop('exists',axis=1)

#and see through the Image path that our Image is in the directory or not 

df['exists'] = df['Image_Path'].map(os.path.exists)

df=df[df['exists']]

len(df)

# so we have all the images present in the directory
df_1=df[df["level"]==1]

df_0=df[df["level"]==0]

print(len(df_1))

print(len(df_0))

df_1=df_1[0:2500]

df_0=df_0[0:2500]

print(len(df_1))

print(len(df_0))

df=pd.concat([df_1,df_0],axis=0)

print(len(df))
img= plt.imread(df.Image_Path.iloc[3])

img.shape
img_list=[]

for i in range(0,len(df)):

    img=plt.imread(df.Image_Path.iloc[i])

    

    img_list.append(img)

X=np.asarray(img_list)

print("Shape of Array",X.shape)

print("Length of Array",len(X))

y=df.level.values

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(

    X,y, test_size=0.10, random_state=42)



del X



y_train = to_categorical(y_train, num_classes=2)

y_test_Categorical=to_categorical(y_test)

from keras.models import Sequential,Model

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,Activation,BatchNormalization

from keras import losses

from keras.optimizers import Adam, Adagrad

from keras.callbacks import EarlyStopping

from keras import regularizers

from sklearn.model_selection import GridSearchCV

import keras

from keras.layers import LeakyReLU

model = Sequential()

model.add(Conv2D(16,kernel_size = (2,2),activity_regularizer = regularizers.l2(1e-6)))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(5, 5),strides=3, padding='same', data_format=None))

model.add(Dropout(0.2))



model.add(Conv2D(32,kernel_size = (2,2)))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(3, 3),strides=2, padding='same', data_format=None))





model.add(Conv2D(64,kernel_size = (2,2)))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2),strides=2, padding='same', data_format=None))

model.add(Dropout(0.4))



model.add(Conv2D(64,kernel_size = (2,2),activity_regularizer = regularizers.l2(1e-6)))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2),strides=2, padding='same', data_format=None))





model.add(Conv2D(128,kernel_size = (2,2)))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2),strides=2, padding='same', data_format=None))

model.add(Conv2D(128,kernel_size = (2,2)))

model.add(LeakyReLU(alpha=0.1))

model.add(Dropout(0.6))









model.add(Flatten())



model.add(Dense(1024,activity_regularizer = regularizers.l2(1e-6)))

model.add(LeakyReLU(alpha=0.1))

model.add(Dropout(0.5))



model.add(Dense(256,activity_regularizer = regularizers.l2(1e-6)))

model.add(LeakyReLU(alpha=0.1))

model.add(Dropout(0.5))



model.add(Dense(128,activity_regularizer = regularizers.l2(1e-6)))

model.add(LeakyReLU(alpha=0.1))

model.add(Dropout(0.3))



model.add(Dense(64,activity_regularizer = regularizers.l2(1e-6)))

model.add(LeakyReLU(alpha=0.1))





#model.add(Dense(16),activity_regularizer = regularizers.l2(1e-6))

#model.add(LeakyReLU(alpha=0.1))

#model.add(Dropout(0.3))

model.add(Dense(2,activation = 'softmax'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=["accuracy"])

model.fit(X_train,y_train, epochs = 1 ,batch_size =16,validation_data=(X_test,y_test_Categorical))

model.summary()
from keras.utils import plot_model

#plot_model(model, to_file='model.png')

x=plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#x=plt.imread(x)

#plt.imsave(x)

history=model.fit(X_train,y_train, epochs = 400 ,batch_size = 16,validation_data=(X_test,y_test_Categorical))





# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])





plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])



plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
from sklearn.metrics import confusion_matrix

prediction=model.predict(X_test)

y_pred=[]

for i in prediction:

    y_pred.append(i.argmax())

y_pred=np.asarray(y_pred)

true_negative,false_positive,false_negative,true_positive=confusion_matrix(y_test, y_pred).ravel()



print("true_negative: ",true_negative)

print("false_positive: ",false_positive)

print("false_negative: ",false_negative)

print("true_positive: ",true_positive)

print("\n\n Accuracy Measures\n\n")



accuracy=(true_negative+true_positive)/(false_negative+false_positive+true_positive+true_negative)

print("Accuracy: ",accuracy)





Sensitivity=true_positive/(true_positive+false_negative)

print("Sensitivity: ",Sensitivity)



False_Positive_Rate=false_positive/(false_positive+true_negative)

print("False_Positive_Rate: ",False_Positive_Rate)



Specificity=true_negative/(false_positive + true_negative)

print("Specificity: ",Specificity)



#FDR à 0 means that very few of our predictions are wrong

False_Discovery_Rate=false_positive/(false_positive+true_positive)

print("False_Discovery_Rate: ",False_Discovery_Rate)



Positive_Predictive_Value =true_positive/(true_positive+false_positive)

print("Positive_Predictive_Value: ",Positive_Predictive_Value)



negative_Predictive_Value =true_negative/(true_positive+false_negative)

print("Negative_Predictive_Value: ",negative_Predictive_Value)
# serialize model to YAML

model_yaml = model.to_yaml()

with open("model.yaml", "w") as yaml_file:

    yaml_file.write(model_yaml)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")

 