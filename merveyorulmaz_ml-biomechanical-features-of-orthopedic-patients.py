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
df = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

df.head()
df.info()
df.describe()
color_list = ['red' if i=="Abnormal" else 'green' for i in df.loc[:,'class']]
pd.plotting.scatter_matrix(df.loc[:, df.columns !='class'],

                                       c=color_list,

                                       figsize=[15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s=200,

                                       marker="*",

                                       edgecolor="black")

plt.show()
data = df[df['class']=="Abnormal"]

x = np.array(data.loc[:,'pelvic_incidence']).reshape(-1,1)

y = np.array(data.loc[:,'sacral_slope']).reshape(-1,1)



#scatter

plt.figure(figsize=[10,10])

plt.scatter(x,y,color="orange")

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
# Linear Regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()



#fit

lr.fit(x,y)



#predicted 

y_head = lr.predict(x)



# scatter

plt.plot(x, y_head, color='red', linewidth=3)

plt.scatter(x,y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
from sklearn.metrics import r2_score

print("RSquare : ",r2_score(y,y_head))