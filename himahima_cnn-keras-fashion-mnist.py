import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")


image_rows = 28
image_cols = 28
batch_size = 4096
image_shape = (image_rows,image_cols,1) 
train.head()
test.head()
train_data = np.array(train, dtype = 'float32')
test_data = np.array(test, dtype='float32')

X_train = train_data[:,1:]/255

y_train = train_data[:,0]

X_test= test_data[:,1:]/255

y_test=test_data[:,0]
from sklearn.model_selection import train_test_split

#Splitting data

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.33, random_state=44, shuffle =True)

class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10, 10))
for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i].reshape((28,28)))
    label_index = int(y_train[i])
    plt.title(class_names[label_index])
plt.show()
X_train = X_train.reshape(X_train.shape[0],*image_shape)
X_test = X_test.reshape(X_test.shape[0],*image_shape)
X_val = X_val.reshape(X_val.shape[0],*image_shape)
import tensorflow as tf
import keras

# STEP 1 : Building the Model 

Model = keras.models.Sequential([
        keras.layers.Conv2D(filters = 32, kernel_size = (4,4),
                            strides = (1,1), padding = 'Same' ,
                            input_shape = image_shape, activation = 'relu' ),
        keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(filters = 64, kernel_size = (4,4),
                            strides = (1,1), padding = 'Same' ,
                            activation = 'relu' ),
        keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation = tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation = tf.nn.softmax),
        ])
# STEP 2 : Compiling the Model 

Model.compile(optimizer ='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Step 3 : Training . .   

history = Model.fit(X_train, y_train, epochs=5)

print('Model Details are : ')
print(Model.summary())


# STEP 4 : Predicting  

y_pred = Model.predict(X_train)

print('Prediction Shape is {}'.format(y_pred.shape))
print('Prediction items are {}'.format(y_pred[:5]))



# STEP 5 : Evaluating   

ModelLoss, ModelAccuracy = Model.evaluate(X_train, y_train)

print('Model Loss is {}'.format(ModelLoss))
print('Model Accuracy is {}'.format(ModelAccuracy ))
ModelLoss, ModelAccuracy = Model.evaluate(X_test, y_test)

print('Model Loss is {}'.format(ModelLoss))
print('Model Accuracy is {}'.format(ModelAccuracy ))