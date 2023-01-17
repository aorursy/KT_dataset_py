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
data = pd.read_csv("/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv")

data.describe()

X = np.array(data.drop(['target'], axis=1))

Y = np.array(data['target'])

print(np.shape(X))

print(np.shape(Y))
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
[X_train, X_test, Y_train, Y_test]= train_test_split(X, Y, test_size=0.1,random_state=1)

classifier = SVC(kernel = 'linear')

model = classifier.fit(X_train, Y_train)

model.predict(X_test)



Classifier2 = LogisticRegression()

model2 = Classifier2.fit(X_train,Y_train)

model2.predict(X_test)


