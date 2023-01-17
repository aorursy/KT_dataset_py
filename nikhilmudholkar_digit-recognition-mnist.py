def convert(imgf, labelf, outf, n):

    f = open(imgf, "rb")

    o = open(outf, "w")

    l = open(labelf, "rb")



    f.read(16)

    l.read(8)

    images = []



    for i in range(n):

        image = [ord(l.read(1))]

        for j in range(28*28):

            image.append(ord(f.read(1)))

        images.append(image)



    for image in images:

        o.write(",".join(str(pix) for pix in image)+"\n")

    f.close()

    o.close()

    l.close()



convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte",

        "mnist_train.csv", 60000)

convert("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte",

        "mnist_test.csv", 10000)
import numpy as np

import pandas as pd
from keras.layers import Dense

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils import to_categorical
from numpy import argmax
df = pd.read_csv('mnist_train.csv', header = None)

print(df.shape)

X_train = df.drop(df.columns[[0]],axis=1).values

print(X_train.shape)

y_train = df.iloc[:,0].values

y_train = to_categorical(y_train)

print(y_train.shape)
num_pixels = 784

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')

X_train = X_train / 255
early_stopping_monitor = EarlyStopping(patience = 3)
n_cols = X_train.shape[1]

model = Sequential()
model.add(Dense(num_pixels,activation = 'relu',input_shape = (n_cols,)))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train,y_train,epochs = 10, callbacks=[early_stopping_monitor])
df_test = pd.read_csv('mnist_test.csv', header = None)

X_test = df_test.drop(df_test.columns[[0]], axis = 1).values

y_test = df_test.iloc[:,0].values

y_test = to_categorical(y_test)



y_preds = model.predict(X_test)

print(np.argmax(y_preds, axis = 1))
model.save('model_file.h5py')
model = load_model('model_file.h5py')



df_test = pd.read_csv('mnist_test.csv', header = None)

X_test = df_test.drop(df_test.columns[[0]], axis = 1).values

y_test = df_test.iloc[:,0].values

y_test = to_categorical(y_test)





def num_recognize(n) :

    ans = model.predict(X_test[n].reshape(-1,784).astype('float32'))

    print(np.argmax(ans, axis = 1))

    temp = X_test[n].reshape([28,28])

    plt.imshow(temp, interpolation = 'nearest',cmap=plt.cm.gray_r)

    #plt.gray()

    plt.savefig('num_fig.pdf')









num_recognize(100)

import matplotlib

matplotlib.use('Agg')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import PyPDF2

import os

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn import datasets





#Here i am working with the MNIST digits recognition dataset, which has 10 classes, the digits 0 through 9!

#A reduced version of the MNIST dataset is one of scikit-learn's included datasets, and that is the one i will use.

#Each sample in this scikit-learn dataset is an 8x8 image representing a handwritten digit.

#Each pixel is represented by an integer in the range 0 to 16, indicating varying levels of black.

#For MNIST dataset, scikit-learn provides an 'images' key in addition to the 'data' and 'target' keys. Because it is a 2D array of the images

#corresponding to each sample, this 'images' key is useful for visualizing the images. On the other hand, the 'data' key contains the feature array

#the images as a flattened array of 64 pixels. I have used the K Nearest Neighbors algorithm









digits = datasets.load_digits()

X = digits.data

y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state = 21)



def learn(k) :

    """this function is used to find the efficiency of algorithm for different values of k and just prints them"""

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train,y_train)

    print(knn.score(X_test, y_test))



def num_predict(i,k) :

    """recognises the number whose data is passed and prints it on console. It also saves the image in the same directory for crosschecking"""

    plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')

    plt.savefig('fig1.pdf')

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train,y_train)

    num = knn.predict([digits.data[i]])

    return num











learn(2) #for checking the efficiency for any values of k

ans = num_predict(1010,7) # here we are passing 2 arguments, the index of data of image that we are trying to predict and k

print("the recognised value is = "+ str(ans))









#the following code draws a line plot between efficiency and different values of k for both testing and training data



neighbors = np.arange(1,20)

train_accuracy = np.empty(len(neighbors))

test_accuracy=np.empty(len(neighbors))



for i,k in enumerate(neighbors) :

    knn = KNeighborsClassifier(k)

    knn.fit(X,y)

    train_accuracy[i] = knn.score(X_train,y_train)

    test_accuracy[i] = knn.score(X_test,y_test)

    plt.clf()





plt.title('k-NN: Varying Number of Neighbors')

plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')

plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')



plt.show(block = True)

plt.savefig('efficiencycomparison.png')



# the image is saved as png file in the working directory
#digit regonition using logistic regression

#first do hyper parameter tuning to fing the best parameters

# then fit the data using those parameters

#draw roc curve

#find area of roc curve

#print confusion matrix and classification_report



import matplotlib

matplotlib.use('Agg')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import PyPDF2

import os

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn import datasets

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV







digits = datasets.load_digits()

X = digits.data

y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state = 21)



#we will do hyperparameter tuning to find the best parameter using gridsearchCV

c_space = np.logspace(-5,8,15)

param_grid= {'C' : c_space, 'penalty':['l1','l2']}

logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg, param_grid, cv = 5)

logreg_cv.fit(X_train,y_train)



print("the best parameter for logistic regression using paramter hypertuning(GridSearchCV) is : " +str(logreg_cv.best_params_))

print("the best score for logistic regression using parameter hypertuning(GridSearchCV) is : "+str(logreg_cv.best_score_))





#the out og this program is

#the best parameter for logistic regression using paramter hypertuning is : {'penalty': 'l1', 'C': 0.051794746792312128}

#the best score for logistic regression using parameter hypertuning is : 0.969769291965

#we will use these parameters for our logistic regression implemented in logistic_regression.py file





#hyperparameter tuning using randomized searchCV

logreg_cv = RandomizedSearchCV(logreg,param_grid,cv = 5)



logreg_cv.fit(X_train,y_train)



print("the best parameter for logistic regression using paramter hypertuning(RandomizedSearchCV) is : " +str(logreg_cv.best_params_))

print("the best score for logistic regression using parameter hypertuning(RandomizedSearchCV) is : "+str(logreg_cv.best_score_))







#digit regonition using logistic regression

#first do hyper parameter tuning to fing the best parameters

# then fit the data using those parameters

#draw roc curve

#find area of roc curve

#print confusion matrix and classification_report



import matplotlib

matplotlib.use('Agg')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import PyPDF2

import os

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn import datasets

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV







digits = datasets.load_digits()

X = digits.data

y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state = 21)

#using parameters as computed by logistic_regression_hypertuning.py

logreg = LogisticRegression(C= 0.051794746792312128, penalty = 'l1')

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

print("the score of logistic regression is : " +str(logreg.score(X_test,y_test)))



print("##### classification report for KNN algorithm")

print("score : "+str(logistic_regression_hypertuning.score(X_test, y_test)))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))




