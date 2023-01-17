import matplotlib.pyplot as plt



import h5py #pip install h5py -- https://www.h5py.org/



#load train

f = h5py.File("../input/mnist-hd5f/train.hdf5", 'r')

train_x, train_y = f['image'][...], f['label'][...]

f.close()



#load test

f = h5py.File("../input/mnist-hd5f/test.hdf5", 'r')

test_x, test_y = f['image'][...], f['label'][...]

f.close()



print("train_x", train_x.shape, train_x.dtype)

print("train_y", train_y.shape, train_y.dtype)



plt.imshow(train_x[0])
