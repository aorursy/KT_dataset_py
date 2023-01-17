import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import keras
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train.shape, y_train.shape, X_test.shape, y_test.shape
X_train[0]
class_labels = ["T-shirt/top","Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",]
plt.imshow(X_train[0])

# plt.imshow(X_train[0], cmap = 'Greys')

plt.show()
plt.figure(figsize = (16,16))



j = 1

for i in np.random.randint(0, 1000, 25):

    plt.subplot(5,5,j)

    j+=1

    plt.imshow(X_train[i], cmap = 'Greys')

    plt.axis('off')

    plt.title(class_labels[y_train[i]] + "=" + str(y_train[i]))

#     plt.title(class_labels[y_train[i]] + "=" + str(y_train[i]), fontsize = 20)
X_train.ndim
X_train_dim = np.expand_dims(X_train, -1)

X_test_dim = np.expand_dims(X_test, -1)
X_train_dim.ndim, X_train_dim.shape, X_train.shape
X_train_scale = X_train_dim / 255

X_test_scale = X_test_dim / 255
from sklearn.model_selection import train_test_split
X_train_model, X_val_model, y_train_model, y_val_model = train_test_split(X_train_scale, y_train, test_size = 0.2, random_state = 2020)
X_train_model.shape, X_val_model.shape, y_train_model.shape, y_val_model.shape
cnn_model = keras.models.Sequential(

    [

        keras.layers.Conv2D(

            filters = 32,

            kernel_size = 3,

            strides = (1,1),

            padding = 'valid',

            activation = 'relu',

            input_shape = [28,28,1]   # 1 is for Black-White Image. 3 will be for RGB images.

        ),

        

        keras.layers.MaxPooling2D(pool_size = (2,2)),

        

        keras.layers.Flatten(),

        

        keras.layers.Dense(units = 128, activation = 'relu'),

        

        keras.layers.Dense(units = 10, activation = 'softmax') # Output Layer

        

        

    ]

)
cnn_model.summary()
cnn_model.compile(

    optimizer = 'adam',

    loss = 'sparse_categorical_crossentropy',

    metrics = ['accuracy']

)
cnn_model.fit(X_train_model, y_train_model, epochs = 10, batch_size = 512, verbose = 1, validation_data=(X_val_model, y_val_model))
X_test_scale[0].ndim
y_pred_1 = cnn_model.predict(np.expand_dims(X_test_scale[0], axis = 0)).round(2)
y_pred_1
np.argmax(y_pred_1)
y_test[0]
y_pred = cnn_model.predict(X_test_scale).round(2)

y_pred
cnn_model.evaluate(X_test_scale, y_test)
plt.figure(figsize = (16,30))



j = 1

for i in np.random.randint(0, 1000, 60):

    plt.subplot(10,6,j)

    j += 1

    plt.imshow(X_test_scale[i].reshape(28,28), cmap = 'Greys') # As we aer using the scaled dataset we have to reshape it back to 28*28.

#     plt.imshow(X_test[i], cmap = 'Greys')

    plt.axis('off')

    plt.title("Actual = {} \n Predicted = {}".format(class_labels[y_test[i]], class_labels[np.argmax(y_pred[i])]) )

    
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, [np.argmax(i) for i in y_pred])
plt.figure(figsize = (16,9))

sns.heatmap(cm, annot = True, fmt = "d", xticklabels = class_labels, yticklabels = class_labels)
from sklearn.metrics import classification_report
cr = classification_report(

    y_test, 

    [ np.argmax(i) for i in y_pred],

    target_names = class_labels,

)
print(cr)
cnn_model.save("Faishon_MNIST_CNN_model.h5")
model_loaded = keras.models.load_model('Faishon_MNIST_CNN_model.h5')
y_pred_saved_model = model_loaded.predict(np.expand_dims(X_test_scale[99], axis = 0)).round(2)

prediction = np.argmax(y_pred_saved_model)

prediction
y_test[99]
print("Actual = {} \nPredicted = {}".format(class_labels[y_test[99]], class_labels[prediction]) )
cnn_model2 = keras.models.Sequential(

    [

        keras.layers.Conv2D(

            filters = 32,

            kernel_size = 3,

            strides = (1,1),

            padding = 'valid',

            activation = 'relu',

            input_shape = [28,28,1]   # 1 is for Black-White Image. 3 will be for RGB images.

        ),

        

        keras.layers.MaxPooling2D(pool_size = (2,2)),

        

        keras.layers.Conv2D(

            filters = 64,

            kernel_size = 3,

            strides = (2,2),

            padding = 'same',

            activation = 'relu',

            input_shape = [28,28,1]   # 1 is for Black-White Image. 3 will be for RGB images.

        ),

        

        keras.layers.MaxPooling2D(pool_size = (2,2)),

        

        keras.layers.Flatten(),

        

        keras.layers.Dense(units = 128, activation = 'relu'),

        

        keras.layers.Dropout(0.25),

        

        keras.layers.Dense(units = 256, activation = 'relu'),

        

        keras.layers.Dropout(0.25),

        

        keras.layers.Dense(units = 128, activation = 'relu'),

        

        keras.layers.Dense(units = 10, activation = 'softmax') # Output Layer

        

        

    ]

)
cnn_model2.summary()
cnn_model2.compile(

    optimizer = 'adam',

    loss = 'sparse_categorical_crossentropy',

    metrics = ['accuracy']

)
cnn_model2.fit(X_train_model, y_train_model, epochs = 20, batch_size = 512, verbose = 1, validation_data=(X_val_model, y_val_model))
cnn_model2.evaluate(X_test_scale, y_test)