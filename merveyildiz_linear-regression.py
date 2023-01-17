# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")



x= df.iloc[:,0].values.reshape(-1,1)

y=df.iloc[:,3].values.reshape(-1,1)
from sklearn.linear_model import LinearRegression



lr= LinearRegression()

lr.fit(x,y)

y_head= lr.predict(x)

plt.scatter(x,y)

plt.show()
from sklearn.metrics import r2_score



print("r square: ", r2_score(y,y_head))