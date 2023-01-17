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
import os 

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression #Yaygın olarak kullanılan açık kaynak kodlu machine learning kütüphanesidir.
df = pd.read_csv("../input/salary/Salary.csv")

df.head()
df.info()
x = df.iloc[:, :-1].values

y = df.iloc[:, 1].values
print(x.shape)

print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.33, random_state = 42)
model = LinearRegression()

model.fit(x_train, y_train)
prediction = model.predict(x_test)

print(prediction)
print(y_test)
plt.scatter(x_train, y_train, color = 'red')

plt.plot(x_train, model.predict(x_train), color = 'blue')

plt.title('Çalışma Süresine göre Maaş Durumu')

plt.xlabel('Çalışma Süresi')

plt.ylabel('Maaş')

plt.show()
plt.scatter(x_test, y_test, color = 'red')

plt.plot(x_train, model.predict(x_train), color = 'blue')

plt.title('Çalışma Süresine göre Maaş Durumu')

plt.xlabel('Çalışma Süresi')

plt.ylabel('Maaş')

plt.show()
print(np.mean((prediction - y_test)**2))

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test,prediction))
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
print("MAPE(Mean absolute percentage error) is: ", np.mean(np.abs((y_test - prediction) / y_test)) * 100)