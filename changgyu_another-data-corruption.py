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




matplotlib.rcParams['figure.figsize'] = (8.0, 20.0)

range_to = 5000

for i in range(16):

    plt.subplot(16, 1, i + 1)

    plt.plot(x1[:range_to, i])

    plt.plot(x2[:range_to, i])