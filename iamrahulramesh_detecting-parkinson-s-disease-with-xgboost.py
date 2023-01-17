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
import numpy as np

import pandas as pd

import sklearn



from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
df = pd.read_csv('../input/parkinsons-data-set/parkinsons.data')

df.head()
features = df.loc[:,df.columns != 'status'].values[:,1:]

labels = df.loc[:, 'status'].values
labels
print(labels[labels ==1].shape[0], labels[labels ==0].shape[0])
scaler = MinMaxScaler((-1,1))



X= scaler.fit_transform(features)

y= labels
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 7)
model = XGBClassifier()

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print((accuracy_score(y_test,y_pred))*100)