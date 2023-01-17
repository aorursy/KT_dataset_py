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
df=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
df.head()
df.info()
df.corr()
plt.figure(figsize=(13,10))

sns.heatmap(df.corr(),annot=True)

plt.title("Correlation Heatmap")

plt.show()
x=df.pelvic_incidence.values.reshape(-1,1)

y=df.sacral_slope.values.reshape(-1,1)

plt.figure(figsize=(8,8))

plt.scatter(x,y)

plt.title("Relation of Pelvic incidence and Sacral slope")

plt.xlabel("Pelvic incidence")

plt.ylabel("Sacral slope")

plt.show()
from sklearn.linear_model import LinearRegression as lr

linreg=lr()

linreg.fit(x,y)
array=np.arange(min(x),max(x),0.001).reshape(-1,1)

yhead=linreg.predict(array)
plt.figure(figsize=(8,8))

plt.scatter(x,y,label="Samples")

plt.plot(array,yhead,label="Predict",color="red")

plt.title("Relation of Pelvic incidence and Sacral slope")

plt.xlabel("Pelvic incidence")

plt.ylabel("Sacral slope")

plt.legend()

plt.show()
print("R-Squared score: ",linreg.score(x, y))