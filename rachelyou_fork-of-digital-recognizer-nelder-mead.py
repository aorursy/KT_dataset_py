# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

labeled_images = pd.read_csv('../input/train.csv')

images = labeled_images.iloc[0:10000,1:]

labels = labeled_images.iloc[0:10000,:1]

train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.7, random_state=0)
import matplotlib.pyplot as plt, matplotlib.image as mpimg

i=1

img=train_images.iloc[i].as_matrix()

img=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train_labels.iloc[i,0])
test_images[test_images>0]=1

train_images[train_images>0]=1



img=train_images.iloc[i].as_matrix().reshape((28,28))

plt.imshow(img,cmap='binary')

plt.title(train_labels.iloc[i])
import matplotlib.pyplot as plt, matplotlib.image as mpimg

i=1

img=train_images.iloc[i].as_matrix()

img=img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train_labels.iloc[i,0])
test_images[test_images>0]=1

train_images[train_images>0]=1



img=train_images.iloc[i].as_matrix().reshape((28,28))

plt.imshow(img,cmap='binary')

plt.title(train_labels.iloc[i])
from sklearn.decomposition import PCA

n_components = 150;

pca = PCA(svd_solver='randomized', n_components=n_components)

pca.fit(train_images)

train_images_pca = pca.transform(train_images)

test_images_pca = pca.transform(test_images)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score



clf = SVC()

clf.fit(train_images_pca, train_labels)



pred = clf.predict(test_images_pca)

accuracy = accuracy_score(test_labels, pred)

print(accuracy)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score



def f(x):

    if x[0] < 0:

        x[0] = 0.0000001

    if x[1] < 0:

        x[1] = 0.0000001

    clf = SVC(kernel='rbf',C=x[0],gamma=x[1],class_weight='balanced')

    clf = clf.fit(train_images_pca, train_labels)

    pred = clf.predict(test_images_pca)

    accuracy = accuracy_score(test_labels, pred)

    print("accuracy")

    print(accuracy)

    return -accuracy



# define constants to be used

alpha = 1.0

gamma = 2.0

beta = 0.5

epsilon = 0.05



# initialize x array and other variables

x = np.array([[10002,1.1],[5050,0.09],[9999,0.02]])

fx = np.array([f(x_) for x_ in x])

fxsort = fx.argsort()

fx = fx[fxsort]

x = x[fxsort]

n = 2

count = 0



# iteration

# reflection

while True:

    count += 1

    xnew = []

    xbar = 1/float(n)*np.sum(x[:n-1,:],axis=0) # centroid

    xr = (1+alpha)*xbar - alpha*x[n]

    fxr = f(xr)

    if fx[0] <= fxr <= fx[n-1]:

        xnew = xr #reflection_accepted = true

        fnew = fxr

        print ("reflection")

    elif fxr <= fx[0]:

        # expansion

        xe = gamma*xr + (1-gamma)*xbar

        fxe = f(xe)

        if fxe < fx[0]:

            xnew = xe # expansion_accepted = true

            fnew = fxe

            print ("expansion")

        else:

            xnew = xr # reflection_accepted = true

            fnew = fxr

            print ("reflection")

    else: # fx[n-1] <= fxr:

        # contraction

        if fx[n] <= fxr:

            # internal contraction

            xc = beta*fx[n]+(1-beta)*xbar

        else:

            # external contraction

            xc = beta*xr + (1-beta)*xbar

        fxc = f(xc)

        if fxc < fx[n-1]:

            xnew = xc

            fnew = fxc

            print ("contraction")

        else: # both reflection vertex and contraction vertex are rejected

            # shrinkage

            for i in range(1,n):

                x[i] = (x[i] + x[0])/2.0

            xnew = (x[n] + x[0])/2.0

            fnew = f(xnew)

            fx = np.array([f(x_) for x_ in x])

            print ("shrinkage")

    x[n] = xnew

    # resort the array

    #fx = np.array([f(x_) for x_ in x])

    fx[n] = fnew

    fxsort = fx.argsort()

    fx = fx[fxsort]

    x = x[fxsort]

    # plot the simplex

    p = plt.Polygon(x, closed=True, fill=False)

    ax = plt.gca()

    ax.add_patch(p)

    Delta = max(1.0,np.sum(np.abs(x[0])**2,axis=-1)**(1./2))

    norm_array = np.array([np.sum(np.abs(x[i] - x[0])**2,axis=-1)**(1./2)

        for i in range(1,n+1)])

    rel_size = 1.0/Delta*max(norm_array)

    print("rel_size")

    print(rel_size)

    if rel_size < epsilon:

        break

    if count > 30:

        break



bestX = x[0]

bestF = fx[0]

print("bestX:")

print(bestX)

print("bestF:")

print(bestF)

print("count:")

print(count)



annotation = "f(%3.8f,%3.8f)=%3.8f" % (bestX[0],bestX[1],bestF)

plt.title("Nelder-Mead SVM")

plt.xlabel("x")

plt.ylabel("y")

plt.plot(bestX[0],bestX[1],'^')

plt.annotate(annotation, (bestX[0],bestX[1]))

ax2 = plt.gca()

ax2.autoscale_view(True,True,True)

plt.show()