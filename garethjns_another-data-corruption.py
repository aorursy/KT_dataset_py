import numpy as np

import scipy.io as sio

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
def load_data(filename):

    mat_data = sio.loadmat(filename)

    data_struct = mat_data['dataStruct']

    return data_struct['data'][0, 0]



data1 = load_data('../input/train_1/1_145_1.mat')

data2 = load_data('../input/train_1/1_1129_0.mat')
def remove_dropouts(x):

    res = np.zeros_like(x)

    c = 0

    for t in range(x.shape[0]):

        if (x[t, :] != np.zeros(x.shape[1])).any():

            res[c] = x[t, :]

            c += 1

    return res[:c, :]



x1 = remove_dropouts(data1)

x2 = remove_dropouts(data2)
sum(sum(x1[1:500,:]))
sum(sum(x2[1:500,:]))
x1[1:10,12]
x2[1:10,12]


def plot2(data1, data2, range_to):

    matplotlib.rcParams['figure.figsize'] = (8.0, 20.0)

    for i in range(16):

        plt.subplot(16, 1, i + 1)

        plt.plot(data1[:range_to, i])

        plt.plot(data2[:range_to, i])

        

plot2(data1,data2+300,240000) # Note shift
plot2(x2,x1+3,100) # Note shift
data1 = load_data('../input/train_1/1_127_1.mat')

data2 = load_data('../input/train_1/1_985_0.mat')

plot2(data1,data2+300,2400000)
# Next pair

data1 = load_data('../input/train_1/1_128_1.mat')

data2 = load_data('../input/train_1/1_986_0.mat')

plot2(data1,data2+300,2400000) 
data1 = load_data('../input/train_1/1_133_1.mat')

data2 = load_data('../input/train_1/1_1015_0.mat')

plot2(data1,data2+300,2400000)
data1 = load_data('../input/train_1/1_134_1.mat')

data2 = load_data('../input/train_1/1_1016_0.mat')

plot2(data1,data2+300,2400000)