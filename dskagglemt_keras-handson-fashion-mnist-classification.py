import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train.shape, y_train.shape
X_test.shape, y_test.shape
X_train[0]
y_train[0]
class_labels = ["T-shirt/top","Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",]
class_labels[9]
plt.imshow(X_train[0], cmap = 'Greys')
plt.show()
plt.figure(figsize = (16,16))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(X_train[i], cmap = 'Greys')
    plt.axis('off')
    plt.title(class_labels[y_train[i]] + "=" + str(y_train[i]), fontsize = 20)
X_train_Scale = X_train / 255
X_test_Scale = X_test / 255
X_train_Scale[0]
# All the values are now rage from 0 to 1
model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape = [28,28]),
        keras.layers.Dense(units = 32, activation = 'relu'),
        keras.layers.Dense(units = 10, activation = 'softmax')
    ]
)
model.summary()
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(X_train_Scale, y_train, epochs = 10)
model.evaluate(X_test_Scale, y_test)
y_pred = model.predict(X_test_Scale)
# Lets verify for first value.
# y_pred[0] 
y_pred[0].round(2)
np.argmax(y_pred[0].round(2)) # Get the index for max value.
print('Actual value    : {}'.format(class_labels[y_test[0]]))
print('Predicted value : {}'.format(class_labels[np.argmax(y_pred[0].round(2))]))
plt.figure(figsize = (16,16))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(X_test[i], cmap = 'Greys')
    plt.axis('off')
    plt.title("Actual = {} \n Predicted = {}".format(class_labels[y_test[i]], class_labels[np.argmax(y_pred[i])]) )
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, [np.argmax(i) for i in y_pred])
plt.figure(figsize = (16,9))
sns.heatmap(cm, annot = True, fmt = "d")
from sklearn.metrics import classification_report
cr = classification_report(
    y_test, 
    [ np.argmax(i) for i in y_pred],
    target_names = class_labels,
)
print(cr)
model.save("Faishon_MNIST_Keras_model.h5")
model_loaded = keras.models.load_model('Faishon_MNIST_Keras_model.h5')
model_loaded.predict(X_test_Scale)
