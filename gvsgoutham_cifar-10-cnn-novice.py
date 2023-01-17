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
import pandas as pd
from PIL import Image
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
#!pip install py7zr
import py7zr
with py7zr.SevenZipFile('/kaggle/input/cifar-10/train.7z', mode='r') as z:
    z.extractall(path='/kaggle/working')
train_labels = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv')
train_images = pd.DataFrame(columns = ['id','pixel_array','label'])

test_labels = pd.read_csv('/kaggle/input/cifar-10/sampleSubmission.csv')
test_images = pd.DataFrame(columns = ['id','pixel_array','label'])
base_path = '/kaggle/working/train/'
r = '.png'
img_a = []
for i in range(0,50000):
    img = Image.open(base_path + str(i+1) + r)
    arr = np.array(img)
    arr = arr.ravel()
    train_images = train_images.append([{ 'id':train_labels['id'].iloc[i],'pixel_array': arr, 'label':train_labels['label'].iloc[i]}])
rm -r /kaggle/working/train
with py7zr.SevenZipFile('/kaggle/input/cifar-10/test.7z', mode='r') as z:
    z.extractall(path='/kaggle/working')
base_path = '/kaggle/working/test/'
r = '.png'
img_a = []
for i in range(0,300000):
    img = Image.open(base_path + str(i+1) + r)
    arr = np.array(img)
    arr = arr.ravel()
    test_images = test_images.append([{ 'id':test_labels['id'].iloc[i],'pixel_array': arr, 'label':test_labels['label'].iloc[i]}])
rm -r /kaggle/working/test
knnc = KNeighborsClassifier(n_neighbors=3,algorithm = 'brute').fit(np.vstack(train_images['pixel_array']),train_images['label'])
test_images['predict_labels'] = knnc.predict(np.vstack(test_images['pixel_array']))
from sklearn.metrics import accuracy_score
accuracy_score(test_images['label'],test_images['predict_labels'])