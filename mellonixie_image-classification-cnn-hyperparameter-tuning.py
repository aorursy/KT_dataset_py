# Libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.layers import Dropout
#importing training dataset
train=pd.read_csv('../input/gtsrb-german-traffic-sign/Train.csv')
X_train=train['Path']
y_train=train.ClassId
train
data_dir = "../input/gtsrb-german-traffic-sign"
train_imgpath= list((data_dir + '/' + str(train.Path[i])) for i in range(len(train.Path)))
for i in range(0,9):
    plt.subplot(331+i)
    seed=np.random.randint(0,39210)
    im = Image.open(train_imgpath[seed])  
    plt.imshow(im)
    
plt.show()
train_data=[]
train_labels=[]


path = "../input/gtsrb-german-traffic-sign/"
for i in range(len(train.Path)):
    image=cv2.imread(train_imgpath[i])
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((28,28))
    train_data.append(np.array(size_image))
    train_labels.append(train.ClassId[i])


X=np.array(train_data)
y=np.array(train_labels)
#Spliting the images into train and validation sets

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.20, random_state=7777)  
X_train = X_train.astype('float32')/255 
X_val = X_val.astype('float32')/255

#Using one hote encoding for the train and validation labels
from keras.utils import to_categorical
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)
def create_model(layers):
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[28, 28, 3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    cnn.add(tf.keras.layers.Flatten())
    
    for i, nodes in enumerate(layers):
        cnn.add(tf.keras.layers.Dense(units=nodes, activation='relu'))
            
    cnn.add(tf.keras.layers.Dense(units=43, activation='softmax'))
    
    cnn.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return cnn

model = KerasClassifier(build_fn=create_model, verbose=1)
layers = [[128],(256, 128),(200, 150, 120)]
param_grid = dict(layers=layers)
grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=1)
grid_results = grid.fit(X_train,y_train, validation_data=(X_val, y_val))
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))
def create_model1():
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[28, 28, 3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=43, activation='softmax'))
    
    cnn.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return cnn

model = KerasClassifier(build_fn = create_model1, verbose = 1)

batch_size = [20,40]
param_grid = dict(batch_size=batch_size)

grid = GridSearchCV(estimator = model, param_grid = param_grid, verbose = 1)
grid_results = grid.fit(X_train,y_train, validation_data=(X_val, y_val))

print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
params = grid_results.cv_results_['params']
for mean,param in zip(means,params):
    print('{0} with: {1}'.format(mean,param))
def create_model2(dropout):
    # create model
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[28, 28, 3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
    cnn.add(Dropout(dropout))
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(Dropout(dropout))
    cnn.add(tf.keras.layers.Dense(units=43, activation='softmax'))
    
    cnn.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return cnn

model = KerasClassifier(build_fn = create_model2, verbose = 1, batch_size=20)

dropout = [0.0, 0.1, 0.2]
param_grid = dict(dropout=dropout)

grid = GridSearchCV(estimator = model, param_grid = param_grid, verbose = 1)
grid_results = grid.fit(X_train,y_train, validation_data=(X_val, y_val))

print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
params = grid_results.cv_results_['params']
for mean,param in zip(means,params):
    print('{0} with: {1}'.format(mean,param))
#Definition of the DNN model

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[28, 28, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn.add(Dropout(0.1))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(Dropout(0.1))
cnn.add(tf.keras.layers.Dense(units=43, activation='softmax'))

# compile the model
cnn.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = cnn.fit(X_train, y_train, batch_size=20, epochs=20,validation_data=(X_val, y_val))
import matplotlib.pyplot as plt

plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
test=pd.read_csv('../input/gtsrb-german-traffic-sign/Test.csv')
X_test=train['Path']
y_test=train.ClassId
data_dir = "../input/gtsrb-german-traffic-sign"
test_imgpath= list((data_dir + '/' + str(test.Path[i])) for i in range(len(test.Path)))
test_data=[]
test_labels=[]


path = "../input/gtsrb-german-traffic-sign/"
for i in range(len(test.Path)):
    image=cv2.imread(test_imgpath[i])
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((28,28))
    test_data.append(np.array(size_image))
    test_labels.append(test.ClassId[i])


X_test=np.array(test_data)
y_test=np.array(test_labels)

X_test = X_test.astype('float32')/255 
#predictions-
pred = cnn.predict_classes(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)