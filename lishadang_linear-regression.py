# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/satandgpa-lr/SATandGPA_LinearRegression.csv")

df.head()
df.shape
df.keys()
x = df['SAT'].values

y = df['GPA'].values

x = np.array(x).reshape(-1,1)

LR = LinearRegression()

LR.fit(x,y)
y_pred = LR.predict(x)

print(x)
print(y)


print(y_pred)
plt.figure(figsize=(30,20))

plt.scatter(x,y)

plt.plot(x,y_pred, color= "blue")

plt.show()