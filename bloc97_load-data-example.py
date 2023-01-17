# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



%pylab inline

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
data_train_file = np.load("/kaggle/input/polyai-ml-a20/data_train.npz")

data_test_file = np.load("/kaggle/input/polyai-ml-a20/data_test.npz")
print(data_train_file.files)

print(data_test_file.files)
data_train_file['labels']
images_train = data_train_file['data']

labels_train = data_train_file['labels']

labels_metadata = data_train_file['metadata'].astype(str)

image0 = images_train[65]

label0 = labels_train[65]
plt.imshow(image0)

plt.show()

print("Class:", label0)

print("Name:", labels_metadata[label0])