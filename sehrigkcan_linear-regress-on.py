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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/student-grade-prediction/student-mat.csv')
print(data.info())
df=data.copy()
df.head()
sns.pairplot(df,kind="reg");
G2=np.array(df.G2.values.reshape(-1,1))
G3=np.array(df.G3.values.reshape(-1,1))

x_train, x_test, y_train, y_test = train_test_split(G1,G3,test_size=0.33,random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)

x_train=np.sort(x_train)
y_train=np.sort(y_train)

plt.scatter(x_train,y_train)
plt.plot(x_test,tahmin,color="red")
plt.show()
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
G2_poly = poly.fit_transform(G2)
lr.fit(G2_poly, G3)
predict = lr.predict(G2_poly)

plt.scatter(G2,G3,color="red")
plt.plot(G2,predict,color="blue")
plt.show()
