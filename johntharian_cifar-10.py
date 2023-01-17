import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import cifar10

(X_train,y_train),(X_test,y_test)=cifar10.load_data()
X_train.dtype
y_train.dtype
print(X_train.shape)
'''size, x_dimension ,y_dimension ,channels(r,g,b)'''
print(X_test.shape)
y_train.shape
y_test.shape
i=1005
plt.imshow(X_train[i])
print(y_train[i])
W_grid=15
L_grid=15

fig,axes=plt.subplots(L_grid,W_grid,figsize=(25,25))
axes=axes.ravel()

n_training=len(X_train)

for i in np.arange(0,L_grid*W_grid):
    index=np.random.randint(0,n_training)
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index])
    axes[i].axis('off')
    
plt.subplots_adjust(hspace=0.4)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
num_cat=10
y_train
y_test
'''Should convert decimal y_train values to binary values before 
feeding it into the neural network'''
import keras
y_train=keras.utils.to_categorical(y_train,num_cat)
y_test=keras.utils.to_categorical(y_test,num_cat)
y_train
y_test
'''normalise the data by dividing with 255'''
X_train=X_train/255
X_test=X_test/255
X_train
'''we need input shape without the size'''
X_train.shape
Input_shape=X_train.shape[1:]
Input_shape
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
cnn_model=Sequential()
cnn_model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',
                     input_shape=Input_shape))
cnn_model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.3))


cnn_model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
cnn_model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.2))


cnn_model.add(Flatten())

cnn_model.add(Dense(units=512,activation='relu'))

cnn_model.add(Dense(units=512,activation='relu'))

cnn_model.add(Dense(units=10,activation='softmax')) #output layer i.e 10 units
cnn_model.compile(loss='categorical_crossentropy',
                   optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])
history=cnn_model.fit(X_train,y_train,batch_size=32,epochs=10,shuffle=True)
evaluation=cnn_model.evaluate(X_test,y_test)

print(f"Test accuracy:{evaluation[1]}")
predicted_class=cnn_model.predict_classes(X_test)
predicted_class
y_test
y_test=y_test.argmax(1)
y_test
def class_convert(classes):
    pred=[]
    for i in predicted_class:
        if i ==0:
            pred.append('Airplanes')
        elif i==1:
            pred.append('Cars')
        elif i==2:
            pred.append('Birds')
        elif i==3:
            pred.append('Cats')
        elif i==4:
            pred.append('Deer')
        elif i==5:
            pred.append('Dogs')
        elif i==6:
            pred.append('Frogs')
        elif i==7:
            pred.append('Horses')
        elif i==8:
            pred.append('Ships')
        elif i==9:
            pred.append('Trucks')
    return pred
pred_class=class_convert(predicted_class)
y_class=class_convert(y_test)
L=7
W=7
fig,axes=plt.subplots(L,W,figsize=(12,12))
axes=axes.ravel()

for i in np.arange(0,L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction={}\n True={}'.format(predicted_class[i],y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=1)
L=7
W=7
fig,axes=plt.subplots(L,W,figsize=(12,12))
axes=axes.ravel()

for i in np.arange(0,L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction={}\n True={}'.format(pred_class[i],y_class[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=2)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,predicted_class)
cm

plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True)
import os

directory=os.path.join(os.getcwd(),'saved_models')

if not os.path.isdir(directory):
    os.makedirs(directory)

model_path=os.path.join(directory,'keras_cifar10_1.h5')
cnn_model.save(model_path)