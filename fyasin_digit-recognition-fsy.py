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
test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')
test.head()
train.head()
testa = test.values

print('testa.shape = ' + str(np.shape(testa)))
traina = train.values

print('traina.shape = ' + str(np.shape(traina)))
traina = traina[:,1:]

print(traina.shape)
train_imgs = np.reshape(traina, (42000,28,28))

print(train_imgs.shape)

train_imgs_norm = np.zeros(np.shape(train_imgs))

for i in range(len(train_imgs[:,0,0])):

    train_imgs_norm[i,:,:] = train_imgs[i,:,:]/np.float(np.max(train_imgs[i,:,:]))

print(np.shape(train_imgs_norm))
from matplotlib import pyplot as plt



plt.imshow(train_imgs_norm[0], cmap='gray')

plt.colorbar()

plt.show()



plt.imshow(train_imgs[0], cmap='gray')

plt.colorbar()

plt.show()
X,Y = np.meshgrid(np.linspace(0,10,1000),np.linspace(0,10,1000))
temp = 1/(1+np.exp(-X))

plt.imshow(temp, cmap='gray',vmin=0.5, vmax=1)

plt.colorbar()

plt.show()
1/1.37