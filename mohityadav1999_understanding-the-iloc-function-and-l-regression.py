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
import pandas as pd

data= ({'a':[1,2,3,4],'b':[3,4,5,6],'c':[12,3,4,5]})
df=pd.DataFrame(data)
df
type(df)
df.iloc[0]
df.iloc[0:-1,0:]
df=pd.read_csv("../input/insurance.csv")
df.keys()

df.info()
x=df.age 
y=df.charges
y.values
x=df.iloc[:,0:1].values
y=df.iloc[:,-1:]
y
from sklearn.linear_model import LinearRegression
ml=LinearRegression()
ml.fit(x,y)
m=ml.coef_
c=ml.intercept_
y_pred=m*x+c
y_pred
y_predict=ml.predict(x)
y_predict
import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.plot(x,y_predict,c="red")
a1=45.5
a2=36.5
plt.scatter(x,y)
plt.plot(x,y_predict,c="red")
c=ml.predict([[a1],[a2]])
plt.scatter([a1,a2],c,color=["green","yellow"])
plt.show