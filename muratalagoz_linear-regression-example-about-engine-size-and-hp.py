# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.model_selection as ms
import sklearn.linear_model as lr
import sklearn.metrics as mt
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')
df.head()

hp = df.horsepower
engSize= df.enginesize


x = engSize
y = hp
x= x.values.reshape(-1,1)






x_train, x_test, y_train, y_test = ms.train_test_split(x,y,test_size=1/3,random_state = 0)
reg = lr.LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

score = mt.r2_score(y_test,y_pred)
print(score)
plt.scatter(x_test,y_test, color = 'b')
plt.plot(x_test,y_pred, color = 'r')
plt.show()
plt.show()