#Import useful package
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#Import data and Inspect
bee_image = pd.read_csv('../input/bee_data.csv')
print(bee_image.head())
print(bee_image.tail())
#Encoding health condition to number
le = preprocessing.LabelEncoder()
bee_image['health'] = le.fit_transform(bee_image['health'])
#Function for read image and resize it
def read_image(file):
    img = skimage.io.imread('../input/bee_imgs/bee_imgs/' + file)
    img = skimage.transform.resize(img, (100,100,3), mode='reflect')
    return img
#Apply and create new column
bee_image['img'] = bee_image['file'].apply(read_image)
#Checking that image have load properly
plt.figure()
plt.imshow(bee_image['img'][200])
plt.colorbar()
plt.grid(False)
#Define X ,y and transform data before sent it to model
X = np.stack(bee_image['img'])
y = bee_image['health'].values
np.unique(y)
#Split data to train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,shuffle=True)
#Split Test Data for validation and test
X_validation = X_test[:500]
y_validation = y_test[:500]
X_test = X_test[500:]
y_test = y_test[500:]

#Inspect all dataset
print('X train shape : ' + str(X_train.shape))
print('y train shape : ' + str(y_train.shape))
print('X test shape : ' + str(X_test.shape))
print('y test shape : ' + str(y_test.shape))
print('X validation shape : ' + str(X_validation.shape))
print('y validation shape : ' + str(y_validation.shape))
#Create fn for training
def train_model(learning_rate=0.00001,epochs=10):
    #Function for print dot instead of show all loss when run
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            print('.', end='')
    #Create neural network with 2 hidden layer (Each layer have 512 node)
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100,100,3)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(6, activation=tf.nn.softmax)
    ])
    
    #Define configure for model
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'
                  ])
    #Create earlystop so model will stop automaticly after it doesn't improve for 25 epochs (code from https://www.tensorflow.org/tutorials/keras/basic_regression)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', 
                                               patience=25)
    #Fit model
    history = model.fit(X_train,y_train,epochs=epochs,validation_data=(X_validation,y_validation),callbacks=[early_stop,PrintDot()],verbose=0,batch_size=100)
    #Plot Validation loss
    plt.figure(1)
    plt.plot(range(len(history.history['val_loss'])),history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Training Model')
    return history,model

history,model = train_model(learning_rate=0.00002,epochs=1000)
[loss, acc] = model.evaluate(X_test,y_test,verbose=0)
#Evaluate with Testing set (model didn't seen this data before)
print("Testing set Accruracy Error:{:7.2f}%".format(acc * 100))



