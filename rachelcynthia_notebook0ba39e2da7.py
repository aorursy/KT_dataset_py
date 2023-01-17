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
        print(os.path.join(dirname,'  ', filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
dataset = pd.read_csv('../input/szeged-weather/weatherHistory.csv')
x = dataset.iloc[:500, 3].values.reshape(-1, 1)
y = dataset.iloc[:500, 5].values.reshape(-1, 1)
x[:6]
y[:6]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
regressor = RandomForestRegressor(n_estimators=20)
regressor.fit(x_train, y_train)
x_grid = np.arange(min(x_train), max(x_train), 0.5).reshape(-1, 1)
plt.scatter(x_train, y_train, color="pink")
plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.title('Humidity Vs Temperature(Training Set)')
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.show()
plt.scatter(x_test, y_test, color="pink")
plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.title('Humidity Vs Temperature(Test Set)')
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.show()