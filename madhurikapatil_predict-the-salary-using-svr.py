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
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/kaggle/input/years-of-experience-and-salary-dataset/Salary_Data.csv')
X =  dataset.iloc[:,0:1].values
y  = dataset.iloc[:,1].values
print(dataset)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

y_pred = regressor.predict(sc_X.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)
print(y_pred)
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red') #orignal
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X)), color = 'blue') #predicted
plt.title('Years VS Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
