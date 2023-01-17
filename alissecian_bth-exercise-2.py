# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

from matplotlib.pyplot import legend
import matplotlib.pyplot as plot

import numpy as np
bmi_life_data = pd.read_csv("/kaggle/input/bmi-and-life-expectancy/bmi_and_life_expectancy.csv")
x_values = bmi_life_data.iloc[:, 2].values
y_values = bmi_life_data.iloc[:, 1].values
x_values = x_values.reshape(-1, 1)
y_values = y_values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.3, random_state=42)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
model = LinearRegression()
model.fit(X_train, y_train)
y_predicted_test = model.predict(X_test)
# Plot outputs
import matplotlib.pyplot as plot
plot.scatter(X_train, y_train, color = 'red')
plot.scatter(X_test, y_test, color = 'blue')
plot.plot(X_test, y_predicted_test, color = 'green')
plot.show()