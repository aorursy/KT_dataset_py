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
import numpy as np
from matplotlib import image
from sklearn import neural_network
from sklearn.feature_extraction import image as skimg

data = pd.read_csv('/kaggle/input/super-ai-image-classification/train/train/train.csv')
data2 = pd.read_csv('/kaggle/input/super-ai-image-classification/val/val/val.csv')
data.info()
data2.info()
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

#img = []
#for i in range(data.shape[0]):
#   imglocation = "/kaggle/input/super-ai-image-classification/train/train/images/{image}".format(image=data['id'][i])
#    imgbuff = image.imread(imglocation)
#    img.append(imgbuff)
#img = np.array(img)
#mlp = MLPClassifier(random_state=1, max_iter=300).fit(img,data['category'].values)
X = np.array([0])
y = np.array([0])
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X, y)

dummy_clf.predict(X)

dummy_clf.score(X, y)