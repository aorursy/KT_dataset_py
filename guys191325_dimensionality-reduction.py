def plot_image(orig,transformed=None):

    plt.subplot(1, 2, 1);

    plt.imshow(orig.reshape(28,28))

    plt.title('Original Image', fontsize = 20);

    if transformed is not None:

        plt.subplot(1, 2, 2);

        plt.imshow(transformed.reshape(28, 28))

        plt.title('Transformed', fontsize = 20);
from sklearn.decomposition import PCA

from keras.datasets import mnist,fashion_mnist

import numpy as np

#(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_w=28

image_h=28

#fashion_mnist = keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_size=60000

test_szie=10000
import matplotlib.pyplot as plt

%matplotlib inline

plot_image(x_train[0])
x_train = x_train.reshape(60000,image_h*image_w)

x_test = x_test.reshape(10000,image_h*image_w)
pca = PCA(5)

pca.fit(x_train)

TRAIN= pca.transform(x_train)

TEST = pca.transform(x_test)

approximation = pca.inverse_transform(TEST)

plot_image(x_test[0],approximation[0])

pca = PCA(2)

pca.fit(x_train)

TRAIN= pca.transform(x_train)

TEST = pca.transform(x_test)

approximation = pca.inverse_transform(TEST)



plots=[]

for y,name in enumerate(class_names):

    new_set = TRAIN

    X=np.array([x for i,x in enumerate(new_set[:, 0]) if y_train[i]==y]).reshape(-1)

    Y=np.array([x for i,x in enumerate(new_set[:, 1]) if y_train[i]==y]).reshape(-1)

    p= plt.scatter(X, Y)

    plots.append(p)

plt.legend(plots,class_names)

plt.show()
pca = PCA(image_w*image_h)

pca.fit(x_train)
pca = PCA(image_h*image_w)

pca.fit(x_train)

plt.plot(pca.explained_variance_ratio_)



plt.plot(range(len(pca.explained_variance_ratio_)),[pca.explained_variance_ratio_.mean()]*len(pca.explained_variance_ratio_), label='Mean', linestyle='--')

plt.xlabel('component')

plt.ylabel('explained variance');



plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');

subset=1000


from sklearn.manifold import TSNE

tsne=TSNE(n_components=2) #n_iter=250

TRAIN = tsne.fit_transform(x_train[:subset])



plots=[]

for y,name in enumerate(class_names):    

    X=np.array([x for i,x in enumerate(TRAIN[:, 0]) if y_train[i]==y])

    Y=np.array([x for i,x in enumerate(TRAIN[:, 1]) if y_train[i]==y])

    p= plt.scatter(X, Y)

    plots.append(p)

plt.legend(plots,class_names)

plt.show()

import umap



umap = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean") #those are deafult params in the package

TRAIN = umap.fit_transform(x_train[:subset])





plots=[]

for y,name in enumerate(class_names):    

    X=np.array([x for i,x in enumerate(TRAIN[:, 0]) if y_train[i]==y])

    Y=np.array([x for i,x in enumerate(TRAIN[:, 1]) if y_train[i]==y])

    p= plt.scatter(X, Y)

    plots.append(p)

plt.legend(plots,class_names)

plt.show()
