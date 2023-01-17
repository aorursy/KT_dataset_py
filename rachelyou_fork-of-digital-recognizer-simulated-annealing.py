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

from sklearn.metrics import accuracy_score



# define the function

def f(x):

    if x[0] < 0:

        x[0] = 0.0000001

    if x[1] < 0:

        x[1] = 0.0000001

    print("x0")

    print(x[0])

    print("x1")

    print(x[1])

    clf = SVC(kernel='rbf',C=x[0],gamma=x[1],class_weight='balanced')

    clf = clf.fit(train_images_pca, train_labels)

    pred = clf.predict(test_images_pca)

    accuracy = accuracy_score(test_labels, pred)

    print("accuracy")

    print(accuracy)

    return -accuracy



# define constants to be used

n = 2

iterations = 5

m = 10

radius = 3.0

sigma = 1.0

alpha = 1.0



# define initial conditions

x = np.zeros((iterations,2))

fx = np.zeros(iterations)

x[0] = [10000.0,0.1]

fx[0] = f(x[0])



#iteration

for k in range(0,iterations-1):

    u = np.zeros((m,n))

    count = 0

    while count < m:

        unew = np.zeros(n)

        #for i in range(0,n):

            #unew[i] = np.random.normal(x[k,i],sigma)

        unew[0] = np.random.normal(x[k,0],sigma)

        unew[1] = np.random.normal(x[k,1],sigma/10.0)

        distance = np.sum(np.abs(unew-x[k])**2,axis=-1)**(1./2)

        if distance < radius:

            u[count] = unew

            count += 1



    fu = np.array([f(u_) for u_ in u])

    j = np.argmin(fu)



    # accept the new variable if function value gets better

    if fu[j] < fx[k]:

        x[k+1] = u[j]

        fx[k+1] = f(x[k+1])

    else:

        p = np.zeros(m)

        for i in range(0,m):

            p[i] = np.exp(alpha*(fx[k]-fu[i]))

        S = np.sum(p)

        p = p/S

        xi = np.random.rand()

        for i in range(0,m):

            if xi < np.sum(p[:i]):

                x[k+1] = u[i]

                fx[k+1] = f(x[k+1])



bestX = x[np.argmin(fx)]

bestF = np.min(fx)



# print the function values changing

print(fx)

print("Best variable result:")

print(bestX)

print("Best function value:")

print(bestF)



annotation = "f(%3.8f,%3.8f)=%3.8f" % (bestX[0],bestX[1],bestF)

# plot

plt.plot(x[:,0],x[:,1])

plt.title("Simulated Annealing SVM")

plt.xlabel("x")

plt.ylabel("y")

plt.plot(bestX[0],bestX[1],'^')

plt.annotate(annotation, (bestX[0],bestX[1]))

plt.show()