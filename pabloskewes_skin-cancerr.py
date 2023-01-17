

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
data = pd.read_csv('../input/skin-cancer-mnist-ham10000/hmnist_28_28_RGB.csv')



y = np.array(data['label'])

X = np.array(data.drop(['label'],axis = 1))

l = len(X)

X = X.reshape(l,28,28,3)



labels = [k for k in range(7)]
data.label.value_counts()
plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(X[i])

    plt.title('Label: %i' % y[i])

    

from keras.utils.np_utils import to_categorical

y_cat = to_categorical(y)

print(y_cat)
# Normalisation entre 0 et 1

X = X / 255

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.1, random_state=1)
# Réseau convolutionnel simple

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(28,28, 3), activation='relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

#model.add(Dense(128, activation='relu'))

model.add(Dense(7, activation='softmax'))



# Compilation du modèle

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])



model.summary()
# Apprentissage

train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=1)
# Test

scores = model.evaluate(X_test, y_test, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
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
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(20, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(20, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(7, activation='softmax'))



model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])



model.summary()
# Apprentissage

train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=200, verbose=1)



# Test

scores = model.evaluate(X_test, y_test, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
plot_scores(train)