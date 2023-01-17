# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import cv2 as cv



files = []



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
len(files)
while len(files)>90380:

    for file in files:

        if '/Training/' not in file and '/Test/' not in file:

            files.remove(file)
len(files)
X = []

y = []

for file in files:

    img = cv.imread(file)

    img = cv.resize(img,(25,25))

    img = img.reshape(1875)

    X.append(img)

    y.append(file.split('/')[-2])
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
X = np.array(X)

y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = X_train/255

X_test = X_test/255
lr = LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc.score(X_test,y_test)