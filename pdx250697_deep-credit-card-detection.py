import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec

from sklearn.manifold import TSNE
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.optimizers import SGD
card = pd.read_csv("F:/creditcard.csv")
X = card.iloc[:,:-1]
y = card['Class']
card.head()
frauds = card.loc[card['Class'] == 1]
non_frauds = card.loc[card['Class'] == 0]
print("We have", len(frauds), "fraud data points and", len(non_frauds), "regular data points.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(("Size of X_training set:"  , X_train.shape))
print(("Size of X_testing set:"  , X_test.shape))
print(("Size of y_training set:"  , y_train.shape))
print(("Size of y_testing set:"  , y_test.shape))
model =Sequential ()
model.add(Dense(30, input_dim=30, activation='relu'))     # kernel_initializer='normal'
model.add(Dense(12, activation='sigmoid'))  
model.add(Dense(output_dim = 1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()
model.fit(X_train.as_matrix(), y_train, epochs=1)
import pandas_ml as pdml
print("Loss: ", model.evaluate(X_test.as_matrix(), y_test, verbose=0))
y_predicted = model.predict(X_test.as_matrix()).T[0].astype(int)
from  pandas_ml import ConfusionMatrix
y_right = np.array(y_test)
confusion_matrix = ConfusionMatrix(y_right, y_predicted)
print("Confusion matrix:\n%s" % confusion_matrix)
confusion_matrix.plot(normalized=True)
plt.show()
confusion_matrix.print_stats()
from  sklearn.decomposition  import PCA
from sklearn.preprocessing import scale
card2 = pdml.ModelFrame(X_train, target=y_train)
sampler = card2.imbalance.over_sampling.SMOTE()
oversampled = card2.fit_sample(sampler)
X2, y2 = oversampled.iloc[:,:-1], oversampled['Class']

data = scale(X2)
pca = PCA(n_components=10)
X2 = pca.fit_transform(data)
X2
model2=Sequential()
model2.add(Dense(10, input_dim=10, activation='relu')) 
model2.add(Dense(27, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(30, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(15, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model2.summary()
X2_test  = pca.fit_transform(X_test)
h = model2.fit(X2, y2, epochs=1, validation_data=(X2_test, y_test),batch_size=100)
print("Loss: ", model2.evaluate(X2_test, y_test, verbose=2))
y2_predicted  = np.round(model2.predict(X2_test)).T[0]
y2_correct = np.array(y_test)
confusion_matrix2 = ConfusionMatrix(y2_correct, y2_predicted)
print("Confusion matrix:\n%s" % confusion_matrix2)
confusion_matrix2.plot(normalized=True)
plt.show()
confusion_matrix2.print_stats()
confusion_matrix2.print_stats()

