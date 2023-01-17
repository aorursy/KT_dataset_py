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
import pandas as pd

data = pd.read_csv("../input/coronavirus_till_date14feb.csv")

import numpy as np

X=data['Infected'].values

Y=data['Recovered'].values

import seaborn as sns

sns.countplot(x="Infected", data=data)

sns.countplot(x="Recovered", data=data)

mean_x=np.mean(X)

mean_y=np.mean(Y)

m=len(X)

num=0

den=0



for i in range(m):

    num+=(X[i]-mean_x)*(Y[i]-mean_y)

    den+= (X[i]-mean_x)*(X[i]-mean_x)

b0=num/den

b1=mean_y-(b0*mean_x)

print(b0,b1)    

s_t=0

s_r=0

m=len(X)

for i in range(m):

   y_pred=b0*X[i]+b1

   s_t+=(y_pred-mean_y)*(y_pred-mean_y)

   s_r+=(Y[i]-mean_y)*(Y[i]-mean_y)



r_square=s_t/s_r

print(r_square)



import matplotlib.pyplot as plt

plt.plot(X,Y)

plt.show()
