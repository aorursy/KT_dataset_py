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
train = pd.read_csv("../input/fashion-mnist_train.csv")

test = pd.read_csv("../input/fashion-mnist_test.csv")
x_train = train.loc[:, 'pixel1':'pixel784'].values.astype('float32')

y_train = train[['label']].values

x_test = test.loc[:, 'pixel1':'pixel784'].values.astype('float32')

y_test = test[['label']].values
x_train /= 255

x_test /= 255

y_train = y_train.ravel()

y_test = y_test.ravel()
print("x_train original shape", x_train.shape)

print("y_train original shape", y_train.shape)
colors = ['black', 'blue', 'violet', 'yellow', 'white', 'red', 'lime', 'aqua', 'orange', 'gray']

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(9, 10))

for i in range(9):

    plt.subplot(3, 3, i+1)

    plt.imshow(x_train.reshape(-1, 28, 28)[i], cmap='gray')

    plt.title(labels[int(y_train[i])])

plt.show()
x_train_subset = x_train[:10000]

y_train_subset = y_train[:10000]
def plot_embedding(x, title=None):

    plt.figure(figsize=(10, 9))

    for i in range(10):

        plt.scatter(x[y_train_subset == i, 0], x[y_train_subset == i, 1], 

                    c=colors[i], label=labels[i], edgecolors='black')

    plt.legend()

    plt.title(title)

    plt.show()
from time import time
from sklearn.decomposition import PCA



t0 = time()

x_train_subset_pca = PCA(n_components=2).fit_transform(x_train_subset)

print("embedding time:", round(time()-t0, 3), "s")



plot_embedding(x_train_subset_pca, "PCA")
from sklearn.manifold import MDS



t0 = time()

x_train_subset_mds = MDS(n_components=2, n_init=1, max_iter=100).fit_transform(x_train_subset)

print("embedding time:", round(time()-t0, 3), "s")



plot_embedding(x_train_subset_mds, "MDS")
from sklearn.manifold import Isomap



t0 = time()

x_train_subset_iso = Isomap(n_neighbors=30, n_components=2).fit_transform(x_train_subset)

print("embedding time:", round(time()-t0, 3), "s")



plot_embedding(x_train_subset_iso, "Isomap")
from sklearn.manifold import LocallyLinearEmbedding



t0 = time()

x_train_subset_lle = LocallyLinearEmbedding(n_neighbors=30, n_components=2, method='standard').fit_transform(x_train_subset)

print("embedding time:", round(time()-t0, 3), "s")



plot_embedding(x_train_subset_lle, "LLE")
from sklearn.manifold import TSNE



t0 = time()

x_train_subset_tsne = TSNE(n_components=2).fit_transform(x_train_subset)

print("embedding time:", round(time()-t0, 3), "s")



plot_embedding(x_train_subset_tsne, "t-SNE")
from sklearn.manifold import SpectralEmbedding



t0 = time()

x_train_subset_se = SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack").fit_transform(x_train_subset)

print("embedding time:", round(time()-t0, 3), "s")



plot_embedding(x_train_subset_se, "Laplacian Eigenmaps")
from sklearn.metrics import accuracy_score
from sklearn import tree



clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)

t0 = time()

clf.fit(x_train, y_train)

print("training time:", round(time()-t0, 3), "s")



t1=time()

y_pred = clf.predict(x_test)

print("predict time:", round(time()-t1, 3), "s")



print(accuracy_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(criterion='entropy', max_depth=10, n_estimators=10)

t0 = time()

clf.fit(x_train, y_train)

print("training time:", round(time()-t0, 3), "s")



t1=time()

y_pred = clf.predict(x_test)

print("predict time:", round(time()-t1, 3), "s")



print(accuracy_score(y_test, y_pred))
from sklearn.naive_bayes import GaussianNB



clf = GaussianNB()

t0 = time()

clf.fit(x_train, y_train)

print("training time:", round(time()-t0, 3), "s")



t1=time()

y_pred = clf.predict(x_test)

print("predict time:", round(time()-t1, 3), "s")



print(accuracy_score(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier



clf = KNeighborsClassifier(n_neighbors=5, p=2, weights='distance')

t0 = time()

clf.fit(x_train, y_train)

print("training time:", round(time()-t0, 3), "s")



t1=time()

y_pred = clf.predict(x_test)

print("predict time:", round(time()-t1, 3), "s")



print(accuracy_score(y_test, y_pred))
from sklearn.svm import SVC



clf = SVC(C=10, kernel='rbf')

t0 = time()

clf.fit(x_train, y_train)

print("training time:", round(time()-t0, 3), "s")



t1=time()

y_pred = clf.predict(x_test)

print("predict time:", round(time()-t1, 3), "s")



print(accuracy_score(y_test, y_pred))
from sklearn.ensemble import AdaBoostClassifier



clf = AdaBoostClassifier(n_estimators=100)

t0 = time()

clf.fit(x_train, y_train)

print("training time:", round(time()-t0, 3), "s")



t1=time()

y_pred = clf.predict(x_test)

print("predict time:", round(time()-t1, 3), "s")



print(accuracy_score(y_test, y_pred))
import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Lambda, Input

from keras.layers import Conv2D, MaxPooling2D

from keras.models import Model

from keras.utils import np_utils

from keras import backend as K
y_train = np_utils.to_categorical(y_train, num_classes=10)

y_test = np_utils.to_categorical(y_test, num_classes=10)
model = Sequential()

model.add(Dense(input_dim=28*28, units=128, activation='relu'))

for i in range(10):

    model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=20)
print(model.evaluate(x_test, y_test)[1])
x_train = x_train.reshape(-1, 28, 28, 1)

x_test = x_test.reshape(-1, 28, 28, 1)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=20)
print(model.evaluate(x_test, y_test)[1])
# from scipy import misc
# x_train = x_train.reshape(-1, 28, 28)

# x_test = x_test.reshape(-1, 28, 28)

# x_train = np.array([misc.imresize(x, (128, 128)) for x in x_train])

# x_test = np.array([misc.imresize(x, (128, 128)) for x in x_test])
# from keras.applications.mobilenet import MobileNet
# input_image = Input(shape=(128, 128))

# input_image_ = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, 3), 3, 3))(input_image)

# base_model = MobileNet(input_shape=(128, 128, 3), input_tensor=input_image_, include_top=False, pooling='avg')

# output = Dropout(0.5)(base_model.output)

# predict = Dense(10, activation='softmax')(output)



# model = Model(inputs=input_image, outputs=predict)

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.summary()
# model.fit(x_train, y_train, batch_size=128, epochs=20)
# print(model.evaluate(x_test, y_test)[1])