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
df=pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")
x = df.iloc[:,0].values.reshape(-1,1)

y = df.iloc[:,3].values.reshape(-1,1)



#visualize

plt.scatter(df.pelvic_incidence,df.sacral_slope)

plt.xlabel("pelvic_incidence")

plt.ylabel("sacral_slope")

plt.show()

from sklearn.linear_model import LinearRegression

reg=LinearRegression()
predict_space= np.linspace(min(x), max(x)).reshape(-1,1)



reg.fit(x,y)
predicted =reg.predict(predict_space)
print("R^2 score: ",reg.score(x,y))
plt.plot(predict_space, predicted, color="black", linewidth = 3)

plt.scatter(x=x, y=y)

plt.xlabel("pelvic_incidence")

plt.ylabel("sacral_slope")

plt.show()
