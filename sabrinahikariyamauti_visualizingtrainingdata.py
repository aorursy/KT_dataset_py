from keras.datasets import mnist
from matplotlib import pyplot
# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
fig = pyplot.figure(figsize=(10,14))
# plot first few images
for i in range(25):
    # define subplot
    ax = fig.add_subplot(5, 5, 1+i,xticks=[],yticks=[])
    ax.set_title(trainy[i])
    # plot raw pixel data
    pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()
from keras.datasets import fashion_mnist
# load dataset
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images
fig = pyplot.figure(figsize=(10,14))
for i in range(50):
    # define subplot
    ax = fig.add_subplot(10, 5, 1+i,xticks=[],yticks=[])
    ax.set_title(trainy[i])
    # plot raw pixel data
    pyplot.imshow(trainX[i])
    #pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()
from keras.datasets import cifar10
# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
for i in range(25):
    # define subplot
    pyplot.subplot(5, 5, 1+i)
    pyplot.xticks([])
    pyplot.yticks([])
    # plot raw pixel data
    pyplot.imshow(trainX[i])
    #pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()
from keras.datasets import cifar100
# load dataset
(trainX, trainy), (testX, testy) = cifar100.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
for i in range(25):
    # define subplot
    pyplot.subplot(5, 5, 1+i)
    pyplot.xticks([])
    pyplot.yticks([])
    # plot raw pixel data
    pyplot.imshow(trainX[i])
    #pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()