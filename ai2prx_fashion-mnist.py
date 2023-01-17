import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
training_set = pd.read_csv('../input/fashion-mnist_train.csv')
test_set = pd.read_csv('../input/fashion-mnist_test.csv')
training_set.head()
training_set.isnull().sum().sort_values(ascending = False)
test_set.isnull().sum().sort_values(ascending = False)
train = np.array(training_set, dtype='float32' )
test = np.array(test_set, dtype='float32')
import random
i = random.randint(1, 60000)
import random
i = random.randint(1, 60000)
plt.imshow(train[i].reshape(28, 28))
label = training_set['label']
W_grid = 15
L_grid = 15

fig, axes = plt.subplots(L_grid, W_grid, figsize = (17, 17))

axes = axes.ravel()

n_training = len(train)

for i in np.arange(0, W_grid * L_grid):
    
    index = np.random.randint(0, n_training)
    
    axes[i].imshow(train[index, 1:].reshape((28, 28)))
    axes[i].set_title(train[index, 0], fontsize = 8)
    axes[i].axis('off')
    
plt.subplots_adjust(hspace=0.4)
X_train = train[:, 1:]/255
y_train = train[:, 0]

X_test = test[:, 1:]/255
y_test = test[:, 0]
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 12345 )
X_train = X_train.reshape(X_train.shape[0], *(28,28,1))
X_val = X_val.reshape(X_val.shape[0], *(28,28,1))
X_test = X_test.reshape(X_test.shape[0], *(28,28,1))
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
cnn_model = Sequential()

cnn_model.add(Conv2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))

cnn_model.add(MaxPooling2D())

cnn_model.add(Flatten())

cnn_model.add(Dense(output_dim = 32, activation = 'relu'))
cnn_model.add(Dense(output_dim = 10, activation = 'sigmoid'))

cnn_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(lr = 0.001), metrics = ['accuracy'])

epochs = 50

cnn_model.fit(X_train, y_train,
             batch_size = 512,
             nb_epoch = epochs, 
             verbose = 1,
             validation_data = (X_val, y_val))
evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))
prediction_classes = cnn_model.predict(X_test)
L = 5
W = 5

fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L * W):
    
    axes[i].imshow(X_test[i].reshape(28, 28))
    axes[i].set_title('Prediction Class = {:0.1f}\n True Class = {:0.1f}'.format(prediction_classes[i], y_test[i]))
    axes[i].axis('off')
    
plt.subplots_adjust(hspace=0.5)
