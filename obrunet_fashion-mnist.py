import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from tensorflow.keras.utils import to_categorical



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
X_train = pd.read_csv('../input/fashion-mnist_train.csv')

X_test = pd.read_csv('../input/fashion-mnist_test.csv')



X_train.shape, X_test.shape
X_train.head()
y_train, y_test = X_train['label'], X_test['label']

X_train, X_test = X_train.iloc[:, 1:], X_test.iloc[:, 1:]

X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train, X_test = np.array(X_train), np.array(X_test)

type(X_train)
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
# display 5 randomly choosen clothes

plt.figure(figsize=(12, 5))



for i in range(1, 5):

    plt.subplot(1, 5, i)

    num = np.random.randint(X_train.shape[0])

    plt.imshow(X_train[num].reshape(28, 28), cmap='gray_r')

    plt.title(class_names[y_train[num]])



plt.show()
# number of clothes for each class

sns.countplot(y_train)
X_train = X_train/255

X_test = X_test/255

X_train.mean(), X_test.mean()
nb_class = len(class_names)



y_train_cat = to_categorical(y_train, num_classes=nb_class, dtype='float32')

y_test_cat  = to_categorical(y_test , num_classes=nb_class, dtype='float32')



y_train.shape, y_train_cat.shape, y_test.shape, y_test_cat.shape
feat_nb = X_train.shape[1]
def create_model(nb_hidden_layers, nb_units):

    model = Sequential()

    

    # 1st layer non hidden

    model.add(Dense(nb_units, input_dim=feat_nb, activation='sigmoid'))

    

    # all hidden layer

    for _ in range(nb_hidden_layers):

        model.add(Dense(nb_units, activation='sigmoid'))

    

    # using softmax for the activiation of the last layer since this is a multi class classification 

    model.add(Dense(nb_class, activation='softmax'))

    

    print(model.summary())

    return model
# creation of a MLP with 2 hidden layers, all layers have 10 units

mlp = create_model(2, 10)
# Compile model

mlp.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
# Fit model 

mlp.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=60, batch_size=32)
loss, accuracy = mlp.evaluate(X_test, y_test_cat)

loss, accuracy
y_pred = mlp.predict(X_test)

y_pred = y_pred.argmax(axis=1)

y_pred
# display 10 randomly choosen clothes with predicted labels and ground truth

plt.figure(figsize=(16, 8))



for i in range(1, 11):

    plt.subplot(2, 5, i)

    num = np.random.randint(X_test.shape[0])

    plt.imshow(X_test[num].reshape(28, 28), cmap='gray_r')

    title = 'truth: ' + class_names[y_test[num]] + '\nprediction: ' + class_names[y_pred[num]]

    plt.title(title)



plt.show()