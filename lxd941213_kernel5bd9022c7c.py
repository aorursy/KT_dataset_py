# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import pandas as pd

import numpy as np

from sklearn.neighbors import KNeighborsClassifier



root_dir = './input/Kannada-MNIST'

train_dir = os.path.join(root_dir, 'train.csv')

test_dir = os.path.join(root_dir, 'test.csv')

train_data = pd.read_csv(train_dir)

labels = train_data.values[:,0]

images = train_data.values[:,1:]





knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(images, labels)



test_data = pd.read_csv(test_dir)

images_test = test_data.values[:,1:]

predictions_knn = knn.predict(images_test)



result = pd.DataFrame({'id':[i for i in range(5000)], 'label': predictions_knn})

result.to_csv('./input/samplesubmission', index=False)