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
metadata = pd.read_csv('../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')
metadata
metadata.dx.value_counts()
df1 = pd.read_csv('../input/skin-cancer-mnist-ham10000/hmnist_28_28_L.csv')
df1.head()
df1.label.value_counts()
labels = ["akiec","bcc", "bkl", "df", "nv", "vasc", "mel"]
y = df1['label']
X = df1.drop(['label'], axis = 1)
X1 = np.array(X)
image = X1[0].reshape(28,28)
n_samples = len(df1.index)
images = X1.reshape(n_samples,28,28)
plt.figure(figsize = (10,20))
for i in range(0,50) :
    plt.subplot(10,5,i+1)
    plt.axis('off')
    plt.imshow(images[i], cmap = 'gray_r')
    plt.title(labels[y[i]])
# On normalise les valeurs entre 0 et 1
X = X/225
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes = (200,100,50))
mlp.fit(X_train,y_train)
y_mlp = mlp.predict(X_test)
mlp_score = accuracy_score(y_test, y_mlp)
print(mlp_score)
Gray_sklearn = mlp_score
pd.crosstab(y_test, y_mlp, rownames = ['Reel'], colnames = ['Prediction'], margins = True)
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
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train = model.fit(X_train , y_train , validation_data=(X_test,y_test), epochs = 50, verbose=1)
model.evaluate(X_test,y_test)
a = model.evaluate(X_test,y_test)
Gray_Keras = a[1]
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
df2 = pd.read_csv('../input/skin-cancer-mnist-ham10000/hmnist_28_28_RGB.csv')
df2.head()
y_ = df2['label']
X_ = df2.drop(['label'], axis = 1)
X1_ = np.array(X_)
image = X1_[0].reshape(28,28,3)
plt.imshow(image)
n_samples = len(df2.index)
images = X1_.reshape(n_samples,28,28,3)
plt.figure(figsize = (10,20))
for i in range(0,50) :
    plt.subplot(10,5,i+1)
    plt.axis('off')
    plt.imshow(images[i])
    plt.title(labels[y[i]])
# On normalise les valeurs entre 0 et 1
X_ = X_/225
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size = 0.2, random_state = 1)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes = (200,100,50))
mlp.fit(X_train,y_train)
y_mlp = mlp.predict(X_test)
mlp_score = accuracy_score(y_test, y_mlp)
print(mlp_score)
Colore_sklearn = mlp_score
pd.crosstab(y_test, y_mlp, rownames = ['Reel'], colnames = ['Prediction'], margins = True)
print(y_[0])
y_cat = to_categorical(y_)
print(y_cat[0])
num_classes = y_cat.shape[1]
print(num_classes)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_, y_cat, test_size = 0.2, random_state = 1)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
model = Sequential()
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train = model.fit(X_train , y_train , validation_data=(X_test,y_test), epochs=30, verbose=1)
model.evaluate(X_test,y_test)
a = model.evaluate(X_test,y_test)
Colore_Keras = a[1]
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
print(Gray_sklearn)
print(Gray_Keras)
print(Colore_sklearn)
print(Colore_Keras)