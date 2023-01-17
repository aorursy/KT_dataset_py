# Directive pour afficher les graphiques dans Jupyter

%matplotlib inline



# Pandas : librairie de manipulation de données

# NumPy : librairie de calcul scientifique

# MatPlotLib : librairie de visualisation et graphiques

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns



from sklearn import model_selection



from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score



from sklearn.preprocessing import StandardScaler, MinMaxScaler



from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split



from sklearn import datasets
from keras.datasets import mnist



from keras.models import Sequential, load_model



from keras.layers import Dense, Dropout, Flatten



from keras.layers.convolutional import Conv2D, MaxPooling2D



from keras.utils.np_utils import to_categorical
df=pd.read_csv('../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')
df.shape
df.head()
df.label.value_counts()
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

print(labels)
n_samples = len(df.index)

images = np.array(df.drop(['label'],axis=1))

images = images.reshape(n_samples,28,28)
plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(images[i], cmap="gray_r")

    plt.title(labels[df.label[i]])
y = df['label']

X = df.drop(['label'] , axis=1)
X = X/255
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(200,60))

mlp.fit(X_train,y_train)

y_mlp = mlp.predict(X_test)
mlp_score = accuracy_score(y_test, y_mlp)

print(mlp_score)
pd.crosstab(y_test, y_mlp, rownames=['Reel'], colnames=['Prediction'], margins=True)
from keras.utils.np_utils import to_categorical
print(y[0])

y_cat = to_categorical(y)

print(y_cat[0])
num_classes = y_cat.shape[1]

print(num_classes)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=1)
X_train = np.array(X_train)

X_test = np.array(X_test)

y_train = np.array(y_train)

y_test = np.array(y_test)
from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

model.add(Dense(200, activation='relu'))

model.add(Dense(60, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train = model.fit(X_train , y_train , validation_data=(X_test,y_test), epochs=30, verbose=1)
model.evaluate(X_test,y_test)
print(train.history['accuracy'])
print(train.history['val_accuracy'])
def plot_scores(train) :

    accuracy = train.history['accuracy']

    val_accuracy = train.history['val_accuracy']

    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')

    plt.plot(epochs, val_accuracy, 'r', label='Score validation')

    plt.title('Scores')

    plt.legend()

    plt.show()
plot_scores(train)
dff=pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
dff.shape
dff.head()
dff.label.value_counts()
labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt",

          "Sneaker","Bag","Ankle boot"]
print(labels[dff.label[0]])
n_samples = len(dff.index)

images = np.array(dff.drop(['label'],axis=1))

images = images.reshape(n_samples,28,28)
plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(images[i], cmap="gray_r")

    plt.title(labels[dff.label[i]])
y = dff['label']

X = dff.drop(['label'] , axis=1)
X = X/255
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(300,50))

mlp.fit(X_train,y_train)

y_mlp = mlp.predict(X_test)
mlp_score = accuracy_score(y_test, y_mlp)

print(mlp_score)
pd.crosstab(y_test, y_mlp, rownames=['Reel'], colnames=['Prediction'], margins=True)
from keras.utils.np_utils import to_categorical
print(y[0])

y_cat = to_categorical(y)

print(y_cat[0])
num_classes = y_cat.shape[1]

print(num_classes)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=1)
X_train = np.array(X_train)

X_test = np.array(X_test)

y_train = np.array(y_train)

y_test = np.array(y_test)
from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

model.add(Dense(200, activation='relu'))

model.add(Dense(60, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train = model.fit(X_train , y_train , validation_data=(X_test,y_test), epochs=30, verbose=1)
model.evaluate(X_test,y_test)
print(train.history['accuracy'])
print(train.history['val_accuracy'])
def plot_scores(train) :

    accuracy = train.history['accuracy']

    val_accuracy = train.history['val_accuracy']

    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')

    plt.plot(epochs, val_accuracy, 'r', label='Score validation')

    plt.title('Scores')

    plt.legend()

    plt.show()
plot_scores(train)