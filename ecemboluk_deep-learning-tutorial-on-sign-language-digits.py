#Add library that need to kernel 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

# filter warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



accuracy_list = [] # accuracy list
x_l =np.load("/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy") # image

y_l = np.load("/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy") # label

img_size = 64

plt.subplot(1,2,1)

plt.imshow(x_l[260].reshape(img_size, img_size))

plt.axis('off')

plt.subplot(1, 2, 2)

plt.imshow(x_l[900].reshape(img_size, img_size))

plt.axis('off')

plt.show()
z = np.zeros(205)

o = np.ones(205)

t = [2 for i in range(205)]

th = [3 for i in range(207)]

f = [4 for i in range(206)]

f5 = [5 for i in range(208)]

s = [6 for i in range(207)]

s7 = [7 for i in range(206)]

e = [8 for i in range(206)]

n = [9 for i in range(207)]

Y = np.concatenate((z, o,t,th,f,f5,s,s7,e,n), axis=0).reshape(x_l.shape[0],1)

print("Y shape:",Y.shape)
X_flatten = x_l.reshape(x_l.shape[0],x_l.shape[1]*x_l.shape[2])

print("X train flatten",X_flatten.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_flatten, Y, test_size=0.2, random_state=42)

print("X_train shape: ",X_train.shape)

print("Y_train shape: ",Y_train.shape)

print("X_test shape: ",X_test.shape)

print("Y_test shape: ",Y_test.shape)
from sklearn import linear_model

logreg = linear_model.LogisticRegression(random_state=42,multi_class ="multinomial",solver="lbfgs")

logreg.fit(X_train, Y_train)

y_head = logreg.predict(X_test)

ac_log = logreg.score(X_test,Y_test)

print("Accuracy: ",ac_log)

accuracy_list.append(ac_log)
from sklearn import metrics 



cnf_matrix = metrics.confusion_matrix(Y_test,y_head.reshape(-1,1)) 

class_names=[0,1,2,3,4,5,6,7,8,9]



fig, ax = plt.subplots(figsize=(10,10))

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="coolwarm" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion Matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
X_flatten = x_l.reshape(x_l.shape[0],x_l.shape[1]*x_l.shape[2])

print("X train flatten",X_flatten.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_flatten, y_l, test_size=0.2, random_state=42)

print("X_train shape: ",X_train.shape)

print("Y_train shape: ",Y_train.shape)

print("X_test shape: ",X_test.shape)

print("Y_test shape: ",Y_test.shape)
# Evaluating the ANN



from keras.models import Sequential #initialize neural network library

from keras.layers import Dense #build our layers library



classifier = Sequential() # initialize neural network

classifier.add(Dense(units = 160, activation = 'relu', input_dim = X_train.shape[1]))

classifier.add(Dense(units = 90, activation = 'relu'))

classifier.add(Dense(units = 80, activation = 'relu'))

classifier.add(Dense(units = 60, activation = 'relu'))

classifier.add(Dense(units = 40, activation = 'relu'))

classifier.add(Dense(units = 20, activation = 'relu'))

classifier.add(Dense(units = 10, activation = 'sigmoid')) #output layer

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model = classifier.fit(X_train,Y_train,epochs=400)

y_head = classifier.predict(X_test)

mean = np.mean(model.history['accuracy'])

accuracy_list.append(mean)

print("Accuracy mean: "+ str(mean))

from sklearn.metrics import confusion_matrix



cnf_matrix= confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(y_head, axis=1))

class_names=[0,1,2,3,4,5,6,7,8,9]



fig, ax = plt.subplots(figsize=(10,10))

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="coolwarm" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion Matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x_l.reshape(-1,64,64,1), y_l, test_size=0.2, random_state=42)

print("X_train shape: ",X_train.shape)

print("Y_train shape: ",Y_train.shape)

print("X_test shape: ",X_test.shape)

print("Y_test shape: ",Y_test.shape)
# Evaluating the CNN



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D



model = Sequential() # initialize neural network

#layer 1

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (64,64,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#layer 2

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

# fully connected

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))





# Compile the model

model.compile(optimizer = "adam" , loss = "categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, Y_train,epochs=200)
y_head = model.predict(X_test)

mean = np.mean(history.history['accuracy'])

accuracy_list.append(mean)

print("Accuracy mean: "+ str(mean))
from sklearn.metrics import confusion_matrix



cnf_matrix= confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(y_head, axis=1))

class_names=[0,1,2,3,4,5,6,7,8,9]



fig, ax = plt.subplots(figsize=(10,10))

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="coolwarm" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion Matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
#Accuracies

column = ["Logistic Regression","Artifical Neural Network","Convolutional Neural Network"]

f,ax = plt.subplots(figsize = (15,7))

sns.barplot(x=accuracy_list,y=column,palette = sns.cubehelix_palette(len(accuracy_list)))

plt.xlabel("Accuracy")

plt.ylabel("Algorithms")

plt.title('Accuracy Values')

plt.show()