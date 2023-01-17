# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data=pd.read_csv('../input/Regression Testing.csv')
data.head()
list(data.columns.values)
X=data['interest rate (%)'].to_frame()
y=data['Median home price'].to_frame()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg=LinearRegression()
reg.fit(X_train,y_train)
#xfit = np.linspace()
y_pred=reg.predict(X_test)
# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title('Regression Training set')
plt.xlabel("Interest Rate")
plt.ylabel("Median price in dollars")
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title('Regression Test set')
plt.xlabel("Interest Rate")
plt.ylabel("Median price in dollars")
plt.show()
