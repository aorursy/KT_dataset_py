import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l1
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout,LayerNormalization
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, SpatialDropout2D, BatchNormalization, LayerNormalization
sns.set(style='white', context='notebook', palette='deep')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()
y=train['label']
y.head()
X=train.drop('label',axis=1)
X.head()
X=X/255
test=test/255
X= X.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.1)
model=Sequential([
    Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu',padding='same'),
    BatchNormalization(),
    SpatialDropout2D(0.25),
    Conv2D(32,(3,3),activation='relu',padding='same'),
    BatchNormalization(),
    SpatialDropout2D(0.25),
    
    Conv2D(32,(3,3),activation='relu',padding='same'),
    BatchNormalization(),
    SpatialDropout2D(0.25),
    Conv2D(32,(5,5),strides=2,activation='relu',padding='same'),
    BatchNormalization(),
    SpatialDropout2D(0.25),
    
    Conv2D(64,(3,3),activation='relu',padding='same'),
    BatchNormalization(),
    SpatialDropout2D(0.25),
    Conv2D(32,(3,3),activation='relu',padding='same'),
    BatchNormalization(),
    SpatialDropout2D(0.25),
    
    
    Conv2D(64,(3,3),activation='relu',padding='same'),
    BatchNormalization(),
    SpatialDropout2D(0.25),
    Conv2D(64,(5,5),strides=2,activation='relu',padding='same'),
    BatchNormalization(),
    SpatialDropout2D(0.25),  
    
    Conv2D(128,(4,4),activation='relu',kernel_regularizer=l1(5e-4),padding='same'),
    Conv2D(8,(3,3),activation='relu',padding='same'),
    BatchNormalization(),
    SpatialDropout2D(0.25),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
])
model.summary()
from tensorflow.keras.optimizers import Adam,RMSprop
opt=RMSprop(
    learning_rate=0.001,
    rho=0.9,
    momentum=0,
    epsilon=1e-08,
    centered=False,
    name="RMSprop"
)

model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X)
history = model.fit_generator(datagen.flow(X,y, batch_size=125),
                              epochs = 30, validation_data = (X_val,y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // 125
                              , callbacks=[learning_rate_reduction])
test_labels=model.predict(test)
test_labels
def plotLearningCurve(history,epochs):
    epochRange = range(1,epochs+1)
    plt.plot(epochRange,history.history['accuracy'])
    plt.plot(epochRange,history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Validation'],loc='upper left')
    plt.show()

    plt.plot(epochRange,history.history['loss'])
    plt.plot(epochRange,history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train','Validation'],loc='upper left')
    plt.show()
plotLearningCurve(history,30)
results = np.argmax(test_labels,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission_final.csv",index=False)
from IPython.display import FileLink
FileLink("submission_final.csv")
results.head()