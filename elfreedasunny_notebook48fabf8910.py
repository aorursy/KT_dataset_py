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

import matplotlib.pyplot as plt

%matplotlib inline
raw_data = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')

test = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')

raw_data.head()
import seaborn as sns

sns.distplot(raw_data['FVC'])
y  = raw_data['FVC']

x =  raw_data[['Patient','Weeks','Percent','Age','Sex','SmokingStatus']]

y_test  = test['FVC']

x_test =  test[['Patient','Weeks','Percent','Age','Sex','SmokingStatus']]

from sklearn.preprocessing import OneHotEncoder

oe = OneHotEncoder()

X = oe.fit_transform(x)

X_test = oe.transform(x_test)
X.shape,X_test.shape
from sklearn.decomposition import TruncatedSVD

tv = TruncatedSVD(n_components = 10)

X_transformed = tv.fit_transform(X)

X_test_transformed = tv.transform(X_test)
X_transformed.shape
X_test_transformed.shape
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters =2)

kmeans.fit(X_transformed,y)

X_transformed= pd.DataFrame(X_transformed)

X_test_transformed1 = pd.DataFrame(X_test_transformed)

X_transformed['labels'] = kmeans.labels_

y_pred_dummy = kmeans.predict(X_test_transformed1)

X_test_transformed1['labels'] =y_pred_dummy
X_transformed
X_test_transformed1
from xgboost import XGBRegressor

xgb = XGBRegressor(reg_lambda = 0.5 ,learning_rate = 1.5)

xgb.fit(X_transformed,y)
xgb.score(X_test_transformed1,y_test)
y_pred = xgb.predict(X_test_transformed1)

y_pred = pd.DataFrame(y_pred)

y_test = pd.DataFrame(y_test)

y_pred
list1 = []

for i in test.index:

    list1.append(test.Patient[i]+str('_-')+str(test.Weeks[i]))

    

list1

sub = pd.DataFrame(columns = ['ID','FVC','Confidence'])

sub['ID'] = list1

sub['FVC'] = y_pred[0]

sub['Confidence'] = 100

sub.to_csv('submission.csv', index=False)
sub