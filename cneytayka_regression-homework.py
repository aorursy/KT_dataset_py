# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
df.head()
df.info()
df.describe()
df.corr()
plt.figure(figsize=(6,8))

sns.heatmap(df.corr(),annot=True,linewidths=.4,fmt=".2f")
plt.show()
df.head()


x=df.pelvic_incidence.values.reshape(-1,1)

y=df.lumbar_lordosis_angle.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("pelvic_incidence")
plt.ylabel("lumbar_lordosis_angle")
plt.show()
from sklearn.linear_model import LinearRegression

linear_reg=LinearRegression()

linear_reg.fit(x,y)

#b0=linear_reg.predict([[50]])
#print(b0)

#b1=linear_reg.coef_
#print(b1)

y_head=linear_reg.predict(x)

plt.plot(x,y_head,color="red")
plt.scatter(x=x,y=y)
plt.show()
from sklearn.metrics import r2_score

print("R2 score:",r2_score(x,y_head))
df.head()
x1=df.pelvic_radius.values.reshape(-1,1)
y2=df.sacral_slope.values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(random_state=42,n_estimators=85)

rf.fit(x1,y2)
x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head2=rf.predict(x_)
plt.scatter(x,y,color="blue")
plt.plot(x_,y_head2,color="purple")
plt.show()