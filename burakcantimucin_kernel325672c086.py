import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('../input/winequality-red.csv')
dataset
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
X.corr()
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
from keras.utils import np_utils

y_train_categorical = np_utils.to_categorical(y_train)
y_test_categorical = np_utils.to_categorical(y_test)
y_train_categorical[0]
X_train

#verilerin olceklenmesi
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
"""
# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

from keras.models import Sequential # initialize neural network library neural networkü initialize etmek için w ve b leri initialize ediyorduk elimizle işte o işlem
from keras.layers import Dense # build our layers library layerları kullanmamızı sağlar
def build_classifier():
    classifier = Sequential() # initialize neural network şuan bir ann yapısı oluşturduk artık layerları eklememiz lazım
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu',input_dim = 11))
    classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier,epochs=500,batch_size=50)
ann=classifier.fit(X_train, y_train_categorical,validation_data=(X_test,y_test_categorical))
ann.history.keys()
y_pred = classifier.predict(X_test)
y_pred
#y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
plt.plot(ann.history['acc'])
plt.plot(ann.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(ann.history['loss'])
plt.plot(ann.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
