#For data

import numpy as np

import pandas as pd
#The model we will use

from sklearn.neighbors import KNeighborsClassifier as KNN
#Used for training set expansion

from scipy.ndimage.interpolation import shift
#For image plotting

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



#For plotting multiple digits

def plot_digits(instances, images_per_row=10, **options):

    size = 28

    images_per_row = min(len(instances), images_per_row)

    images = [instance.reshape(size,size) for instance in instances]

    n_rows = (len(instances) - 1) // images_per_row + 1

    row_images = []

    n_empty = n_rows * images_per_row - len(instances)

    images.append(np.zeros((size, size * n_empty)))

    for row in range(n_rows):

        rimages = images[row * images_per_row : (row + 1) * images_per_row]

        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)

    plt.imshow(image, cmap = mpl.cm.binary, **options)

    plt.axis("off")
#Read our data as a pandas dataframe, initially

mnist = pd.read_csv("../input/train.csv")
#A look at the image data

mnist.head()
#Convert the dataframe to a numpy array (matrix)

mnist = np.array(mnist)
#Split data into predictor and target variables

X, y = mnist[:,1:], mnist[:,0]
img1 = X[1000].reshape(1,784)

img2 = shift(X[1000].reshape(28,28), [0,5], cval=0).reshape(1,784)

plot_digits([img1, img2], images_per_row=1)
img1 = X[1001].reshape(1,784)

img2 = shift(X[1001].reshape(28,28), [0,-5], cval=0).reshape(1,784)

plot_digits([img1, img2], images_per_row=1)
img1 = X[2002].reshape(1,784)

img2 = shift(X[2002].reshape(28,28), [-5,0], cval=0).reshape(1,784)

plot_digits([img1, img2], images_per_row=2)
img1 = X[1234].reshape(1,784)

img2 = shift(X[1234].reshape(28,28), [5,0], cval=0).reshape(1,784)

plot_digits([img1, img2], images_per_row=2)
#---Shifted Sets---

#Will contain all of the right-shifted images

right = np.zeros(shape=(len(X),len(X[0])), dtype="int64")

#Will contain all of the left-shifted images

left = np.zeros(shape=(len(X),len(X[0])), dtype="int64")

#Will contain all of the up-shifted images

up = np.zeros(shape=(len(X),len(X[0])), dtype="int64")

#Will contain all of the down-shifted images

down = np.zeros(shape=(len(X),len(X[0])), dtype="int64")



#For each image in the training set...

for i in range(len(X)):

    #create right-shifted image

    r = shift(X[i].reshape(28,28), [0,1], cval=0).reshape(1,784)

    #create left-shifted image

    l = shift(X[i].reshape(28,28), [0,-1], cval=0).reshape(1,784)

    #create up-shifted image

    u = shift(X[i].reshape(28,28), [-1,0], cval=0).reshape(1,784)

    #create down-shifted image

    d = shift(X[i].reshape(28,28), [1,0], cval=0).reshape(1,784)

    

    #Add shifted images to the shifted sets

    left[i] = l

    right[i] = r

    up[i] = u

    down[i] = d

    

#Append the new data:

Exp_X = np.copy(X)#original data

Exp_X = np.append(Exp_X, left, axis=0)#left-shifted

Exp_X = np.append(Exp_X, right, axis=0)#right-shifted

Exp_X = np.append(Exp_X, up, axis=0)#up-shifted

Exp_X = np.append(Exp_X, down, axis=0)#down-shifted



#Target training data must also be expanded because

#predictor training set is now bigger

Exp_y = np.copy(y)

Exp_y = np.append(Exp_y, y, axis=0)

Exp_y = np.append(Exp_y, y, axis=0)

Exp_y = np.append(Exp_y, y, axis=0)

Exp_y = np.append(Exp_y, y, axis=0)
#initialize the model with pre-selected hyper-parameters

knn_clf = KNN(n_neighbors=4, weights="distance", n_jobs=-1)
#Fit the Model on the expanded training set

knn_clf.fit(Exp_X,Exp_y)
#Read test data

test = pd.read_csv("../input/test.csv")

#convert it to numpy array format

test = np.array(test)
#predictions

predictions = knn_clf.predict(test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("submission.csv", index=False, header=True)