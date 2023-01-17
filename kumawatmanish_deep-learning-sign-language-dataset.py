import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls","../input/Sign-language-digits-dataset"]).decode('utf8'))

X_1 = np.load('../input/Sign-language-digits-dataset/X.npy')
Y_1 = np.load('../input/Sign-language-digits-dataset/Y.npy')
img_size=64
plt.subplot(1,2,1)
plt.imshow(X_1[260].reshape(img_size,img_size))
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(X_1[900].reshape(img_size,img_size))
plt.axis('off')
# Join a sequence of arrays along an row axis.

X=np.concatenate((X_1[204:409],X_1[822:1027]),axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
z= np.zeros(205)
o= np.ones(205)

Y= np.concatenate((z,o),axis=0).reshape(X.shape[0],1)
print('X shape: ',X.shape)
print('Y shape: ',Y.shape)
# Then lets create x_train, y_train, x_test, y_test arrays
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.15, random_state=0)
num_of_train= X_train.shape[0]
num_of_test= X_test.shape[0]
X_train_flatten = X_train.reshape(num_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(num_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)
X_train=X_train_flatten
X_test=X_test_flatten
print(X_train.shape)

#Test over Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)
print(classifier.score(X_test,Y_test))
print(classifier.score(X_train,Y_train))

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def get_classifier():
    k_classifier = Sequential()
    k_classifier.add(Dense(units= 8, activation='relu', kernel_initializer  = 'uniform', input_dim=X_train.shape[1]))
    k_classifier.add(Dense(units= 4, activation='relu', kernel_initializer  = 'uniform' ))
    k_classifier.add(Dense(units= 1, activation='sigmoid', kernel_initializer  = 'uniform'))
    k_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return k_classifier

classifier= KerasClassifier(build_fn= get_classifier, epochs=100)

accuracies= cross_val_score(estimator= classifier, X=X_train, y=Y_train, cv=3)

mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))
