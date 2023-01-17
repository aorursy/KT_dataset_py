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
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df
df.shape
import pandas_profiling

Report = pandas_profiling.ProfileReport(df)
Report
df['diagnosis'] = np.where(df['diagnosis'] == 'M', '1', df['diagnosis'])
df['diagnosis'] = np.where(df['diagnosis'] == 'B', '0', df['diagnosis'])
df
df['diagnosis']
import matplotlib.pyplot as plt
plt.scatter(df['id'],df['radius_mean'])
plt.plot(df['id'],df['radius_mean'])
plt.show()
import matplotlib.pyplot as plt
plt.scatter(df['id'],df['texture_mean'])
plt.plot(df['id'],df['texture_mean'])
plt.show()
import matplotlib.pyplot as plt
plt.scatter(df['id'],df['perimeter_mean'])
plt.plot(df['id'],df['perimeter_mean'])
plt.show()
import matplotlib.pyplot as plt
plt.scatter(df['radius_mean'],df['perimeter_mean'])
plt.plot(df['radius_mean'],df['perimeter_mean'])
plt.xlabel('radius_mean')
plt.ylabel('perimeter_mean')
plt.show()
X = df['radius_mean'].values
X
Y = df['perimeter_mean'].values
Y
X = X.reshape(-1, 1)
X

from sklearn import linear_model
from sklearn.linear_model import LinearRegression 
linear = LinearRegression()
linear.fit(X,Y)
x_predict = linear.predict(X)
x_predict
import matplotlib.pyplot as plt
plt.scatter(X,Y)
plt.plot(X,Y, color='blue')
plt.plot(X, x_predict, color='red')
plt.show()
Slop = linear.coef_
Slop 
Intercept = linear.intercept_
Intercept
linear.score(X,Y)
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing 
encoder = preprocessing.LabelEncoder()
encoder
Y_encoded = encoder.fit_transform(Y)
Y_encoded
logistic = LogisticRegression()
logistic.fit(X, Y_encoded)

x_predict = logistic.predict(X)
x_predict
logistic.score(X,x_predict)
import sklearn as sns
from sklearn import model_selection
from sklearn.model_selection import train_test_split
df.head(2)
features = df[['radius_mean' , 'texture_mean' , 'perimeter_mean' , 'area_mean' , 'smoothness_mean' , 'compactness_mean' , 'concavity_mean' , 'concave points_mean' , 'texture_worst']]
features
label = df['diagnosis']
label
x_test , x_train, y_test, y_train = train_test_split(features, label, test_size=0.3)
x_test
y_test
x_train
y_train
x_test.shape
x_train.shape
y_test.shape
y_train.shape
from sklearn import ensemble 
from sklearn.ensemble import RandomForestClassifier 

RFC = RandomForestClassifier()

RFC.fit(x_train, y_train)
RFC
y_predict = RFC.predict(x_test)
y_predict
from sklearn import metrics

RFCAccuracy = metrics.accuracy_score(y_test, y_predict)

RFCAccuracy