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
df= pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
# taking care of categorical values

data = pd.get_dummies(df)
data.info()
# using pca on this dataset

data = data.drop('Attrition_Yes',axis =1)

from sklearn.decomposition import PCA

pca= PCA()

g = pca.fit(data)

#print(g.explained_variance_ratio_)

np.cumsum(g.explained_variance_ratio_)
data_X = data.drop('Attrition_No',axis=1)

data_Y = data['Attrition_No']
data_X = pca.fit_transform(data)
print(len(data))
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(data_X,data_Y,test_size = 0.3)
# classification

from sklearn.svm import SVC

svc = SVC()

svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)

from sklearn import metrics

print(metrics.accuracy_score(y_pred,y_test))
from sklearn.linear_model import LinearRegression

lr= LinearRegression()

g = lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

metrics.r2_score(y_pred,y_test)
#checking what would happen if pca was not applied

data2 = pd.get_dummies(df)
data2 = data2.drop('Attrition_Yes',axis =1 )

data2.head()
data2X = data2.drop('Attrition_No',axis=1)

data2Y = data2['Attrition_No']
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(data2X,data2Y,test_size = 0.3)
# classification

from sklearn.svm import SVC

svc = SVC()

svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)

from sklearn import metrics

print(metrics.accuracy_score(y_pred,y_test))
from sklearn.linear_model import LinearRegression

lr= LinearRegression()

g = lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

metrics.r2_score(y_pred,y_test)