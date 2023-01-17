# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#data reading
df= pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
#EDA
df.head()
df.info()
df.describe()
plt.scatter(df["free sulfur dioxide"],df["total sulfur dioxide"])
plt.xlabel("free sulfur dioxide")
plt.ylabel("total sulfur dioxide")
plt.show()
#Linear regression by sklaern library

from sklearn.linear_model import LinearRegression
lr= LinearRegression()

x=df["free sulfur dioxide"].values.reshape(-1,1)
y=df["total sulfur dioxide"].values.reshape(-1,1)

lr.fit(x,y)

b0=lr.predict([[0]])
print("b0:",b0)

b1=lr.coef_
print("b1:",b1)

#visualize fitted line 
plt.scatter(x,y,color="green")
y_head=lr.predict(x)
plt.plot(x,y_head,color="red")
plt.show()


#R square value of linear regression model

from sklearn.metrics import r2_score
print("r_square_score:",r2_score(y,y_head) )
# Multiple linear regression
from sklearn.linear_model import LinearRegression
multiple_lr= LinearRegression()

x=df.iloc[:,[1,4]].values
y=df["quality"].values.reshape(-1,1)

multiple_lr.fit(x,y)
print("b0:",multiple_lr.intercept_)
print("b1,b2:",multiple_lr.coef_)

multiple_lr.predict(np.array([[0.5,0.8],[1.5,0.6]]))

#R square value of multiple linear regression model

y_head=multiple_lr.predict(x)

from sklearn.metrics import r2_score
print("r_square_score:",r2_score(y,y_head) )