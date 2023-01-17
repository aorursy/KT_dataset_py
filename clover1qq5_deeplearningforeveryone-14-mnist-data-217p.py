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
from keras.datasets import mnist

from keras.utils import np_utils



import sys

import tensorflow as tf
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
import matplotlib.pyplot as plt



import tensorflow as tf
seed = 0 

np.random.seed(seed)

tf.random.set_seed(3)
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print("학습셋 이미지 수: %d 개" % (X_train.shape[0]))

print("테스트셋 이미지 수: %d 개" % (X_test.shape[0]))
import matplotlib.pyplot as plt

plt.imshow(X_train[0], cmap='Greys')

plt.show()
for x in X_train[0]:

    for i in x:

        sys.stdout.write('%d\t' %i)

    sys.stdout.write('\n')

X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255

X_test 
print("class: %d"%(Y_train[0])) #책이랑 다르다. Y_Class_trian은 정의한 적이 없다. Y_train으로 바꿔준다. 
Y_train = np_utils.to_categorical(Y_train, 10) #책이랑 다르다. Y_Class_trian은 정의한 적이 없다. Y_train으로 바꿔준다. 

Y_test = np_utils.to_categorical(Y_test, 10) #책이랑 다르다. Y_Class_test는 정의한 적이 없다. Y_test으로 바꿔준다. 
print(Y_train[0])