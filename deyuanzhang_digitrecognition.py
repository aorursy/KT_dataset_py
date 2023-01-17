from sklearn.datasets import load_digits
digits = load_digits()
print(digits)
print(digits.DESCR)
X = digits.data/16.
Y = digits.target
print(X)
print(X.shape)
print(Y)
print(Y.shape)

print(X.min(), X.max())
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=4)

print(X_train.shape)
print(Y_train.shape)

print(X_test.shape)
print(Y_test.shape)
#input the Y_train.shape, X_test.shape, Y_test.shape here


%matplotlib inline

from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

def loadImg(fpath, size=[8, 8]):
    img = imread(fpath, as_gray=True)
    img = resize(img, size)
    return img

def showImg(img):
    plt.imshow(img, cmap='gray')
    plt.show()
# get the 0th training data
imgtrain = X[2]
# reshape the 64 dimenstion into 8*8
img = imgtrain.reshape(8,8)
# show it
showImg(img)
#shows the label of the 0th training data
print(Y[3])

for i in range(10):
    # get the 0th training data
    imgtrain = X[i]
    # reshape the 64 dimenstion into 8*8
    img = imgtrain.reshape(8,8)
    # show it
    showImg(img)
    #shows the label of the 0th training data
    print(Y[i])

#input your codes here
#1 import nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
#2 declare a NN model
model = KNeighborsClassifier(n_neighbors = 3)
#3 model.fit
model.fit(X_train, Y_train)

#4. model.predict
Y_prediction = model.predict(X_test)
#5 compute accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_test, Y_prediction)
print(acc)


imgdemo = loadImg('../input/digittest/digit/0.jpg') # input the image file path here.
print(imgdemo.shape)
print(imgdemo)
print(imgdemo.max())
showImg(imgdemo)
print(imgdemo.shape)
imgf = imgdemo.flatten()
print(imgf.shape)
result = model.predict([imgf])
print(result)
imgdemo = loadImg('../input/digittest/digit/9.jpg') # input the image file path here.
showImg(imgdemo)
imgf = imgdemo.flatten()
result = model.predict([imgf])
print(result)

from sklearn.datasets import fetch_openml
X, Y = fetch_openml('mnist_784', return_X_y=True)
print(X.shape)
print(Y.shape)
import numpy as np
np.save('mnistX', X)
np.save('mnistY', Y)
import numpy as np
X = np.load('mnistX.npy')
Y = np.load('mnistY.npy', allow_pickle=True)
print(X.shape)
print(Y.shape)

imgtrain = X[0]
img = imgtrain.reshape((28, 28))
showImg(img)
print(Y[0])
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=4)

print(X_train.shape)
print(X_test.shape)

#input your codes here
#1 import nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
#2 declare a NN model
model = KNeighborsClassifier(n_neighbors = 1)
#3 model.fit
model.fit(X_train, Y_train)

#4. model.predict
Y_prediction = model.predict(X_test)
#5 compute accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_test, Y_prediction)
print(acc)
from sklearn.neighbors import KNeighborsClassifier
#2 declare a NN model
model = KNeighborsClassifier(n_neighbors = 1)
#3 model.fit
model.fit(X_train, Y_train)
print(X_train.max())
imgdemo = loadImg('../input/digittest/digit/9.jpg', [28, 28]) # input the image file path here.
showImg(imgdemo)
imgf = imgdemo.flatten()*255.0
print(imgf.max())
result = model.predict([imgf])
print(result)
