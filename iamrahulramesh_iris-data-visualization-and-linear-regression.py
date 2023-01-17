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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
data = pd.read_csv('../input/iris/Iris.csv')

data.head()
print(data.shape)
data.info()
fig =sns.FacetGrid(data,hue='Species',size = 5)

fig.map(plt.scatter,'SepalLengthCm','SepalWidthCm').add_legend()

sns.pairplot(data,hue='Species')


X = data.iloc[:, :-1].values #   X -> Feature Variables

y = data.iloc[:, -1].values #y ->  Target



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

Y_train= le.fit_transform(y)



print(Y_train)
X_train, X_test, y_train, y_test = train_test_split(X, Y_train, test_size = 0.3, random_state = 0)


LR = LinearRegression()

LR.fit(X_train, y_train)



Y_pred = LR.predict(X_test)
from sklearn import metrics
print('y - intercept   :' ,LR.intercept_)
print('beta - coefficient  :',LR.coef_)
print('mean Abs error   :',metrics.mean_absolute_error(y_test,Y_pred))
print('Mean squareroot error    :', metrics.mean_squared_error(y_test,Y_pred))
print('Root Mean Square error RMSE  :',np.sqrt(metrics.mean_squared_error(y_test,Y_pred)))
print (' r squared value  : ', metrics.r2_score(y_test,Y_pred))