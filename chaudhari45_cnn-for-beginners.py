# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib.pyplot as plt # plotting library

%matplotlib inline





from keras.models import Sequential

from keras.layers import Dense , Activation, Dropout

from keras.optimizers import Adam ,RMSprop

from keras import  backend as K

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from keras.datasets import mnist





# load dataset

(x_train, y_train),(x_test, y_test) = mnist.load_data()







# count the number of unique train labels

unique, counts = np.unique(y_train, return_counts=True)

print("Train labels: ", dict(zip(unique, counts)))







# count the number of unique test labels

unique, counts = np.unique(y_test, return_counts=True)

print("\nTest labels: ", dict(zip(unique, counts)))

# sample 25 mnist digits from train dataset

indexes = np.random.randint(0, x_train.shape[0], size=25)

images = x_train[indexes]

labels = y_train[indexes]





# plot the 25 mnist digits

plt.figure(figsize=(5,5))

for i in range(len(indexes)):

    plt.subplot(5, 5, i + 1)

    image = images[i]

    plt.imshow(image, cmap='gray')

    plt.axis('off')

    

plt.show()

plt.savefig("mnist-samples.png")

plt.close('all')