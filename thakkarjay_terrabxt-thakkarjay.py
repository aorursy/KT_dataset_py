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

import seaborn as sns

import numpy as np
Data = pd.read_csv('/kaggle/input/into-the-future/train.csv')
Data.info()
Data.head()
Data.isnull().any()
X = Data['feature_1'].values.reshape(-1,1)

Y = Data['feature_2'].values.reshape(-1,1)
import matplotlib.pyplot as plt



plt.scatter(X,Y)
sns.distplot(Data['feature_1'])
sns.distplot(Data['feature_2'])
sns.jointplot(x='feature_1',y='feature_2',data=Data)
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
model = LinearRegression()

model.fit(X,Y)
Test = pd.read_csv('/kaggle/input/into-the-future/test.csv')

Test.head()

test = Test['feature_1'].values.reshape(-1,1)

i = Test['id'].values.reshape(-1,1)
pred = model.predict(test)

pred = pd.DataFrame({'id': i.flatten(),'feature_2': pred.flatten()})

pred.to_csv('Sub.csv')
model_s =SVR(kernel='linear')
model_s.fit(X,Y)
preds = model_s.predict(test)

preds = pd.DataFrame({'id': i.flatten(),'feature_2': preds.flatten()})

preds.to_csv('Sub1.csv')
print(pred)
print(preds)