import matplotlib.pyplot as plt

import numpy as np



# load data

path = '/kaggle/input/'

# mnist or gauss

m_x = np.loadtxt(fname='%s_x'%(path+'mnist'), delimiter=' ')

m_y = np.loadtxt(fname='%s_y'%(path+'mnist'), delimiter=' ')

g_x = np.loadtxt(fname='%s_x'%(path+'gauss'), delimiter=' ')

g_y = np.loadtxt(fname='%s_y'%(path+'gauss'), delimiter=' ')



# show data

## mnist

data = np.reshape(np.array(m_x[0], dtype=int), [28, 28])

print(data)



## gauss

plt.scatter(g_x[:, 0], g_x[:, 1], c=g_y)

plt.show()
# split data

thredhold = int(len(m_x) * 0.8)

train_data = {'x': m_x[:thredhold], 'y': m_y[:thredhold]}

test_data = {'x': m_x[thredhold:], 'y': m_y[thredhold:]}
class KNN():

    def __init__(self, k, label_num):

        self.k = k

        self.label_num = label_num



    def fit(self, train_data):

        self.train_data = train_data

        # from scipy import spatial

        # self.kdtree = spatial.KDTree(data=self.train_data['x'])#



    def predict(self, test_x):

        predicted_test_labels = np.zeros(shape=[len(test_x)], dtype=int)

        for x, i in zip(test_x, np.arange(len(test_x))):

            predicted_test_labels[i] = self.get_label(x)

        return predicted_test_labels



    def get_label(self, x):

        knn_indexes = self.get_knn_indexes(x)

        label_statistic = np.zeros(shape=[self.label_num])

        for index in knn_indexes:

            label = int(self.train_data['y'][index])

            label_statistic[label] += 1

        return np.argmax(label_statistic)



    def get_knn_indexes(self, x):

        # return self.kdtree.query(x, k=self.k)[1]

        dis = list(map(lambda a: self.distance(a, x), self.train_data['x']))

        knn_pairs = sorted(zip(dis, np.arange(len(dis))), key=lambda x: x[0])[:self.k]

        knn_indexes = [p[1] for p in knn_pairs]

        return knn_indexes



    def distance(self, a, b):

        return np.sqrt(np.sum(np.square(a-b)))
for k in range(1, 10):

    # fit knn

    knn = KNN(k, label_num=10)

    knn.fit(train_data)

    predicted_labels = knn.predict(test_data['x'])



    # evaluate

    accuracy = np.mean(np.equal(predicted_labels, test_data['y']))

    print('k: %2d, accuracy: %.3f' % (k, accuracy))
from sklearn.neighbors import KNeighborsClassifier

from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt

import numpy as np



x = g_x

y = g_y



step = 0.02

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1

y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

grid_data = np.c_[xx.ravel(), yy.ravel()]



knn1 = KNeighborsClassifier(n_neighbors=10)

knn1.fit(x, y)

z1 = knn1.predict(grid_data)



knn2 = KNeighborsClassifier(n_neighbors=1)

knn2.fit(x, y)

z2 = knn2.predict(grid_data)



cmap_light = ListedColormap(['#FF9999', '#AAFFAA'])

cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

fig = plt.figure(figsize=(16,6))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)

ax1.pcolormesh(xx, yy, z1.reshape(xx.shape), cmap=cmap_light)

ax1.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold)

ax2.pcolormesh(xx, yy, z2.reshape(xx.shape), cmap=cmap_light)

ax2.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold)

plt.show()