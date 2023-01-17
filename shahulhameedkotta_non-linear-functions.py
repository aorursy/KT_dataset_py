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

import matplotlib.pyplot as plt

%matplotlib inline
x = np.arange(-5.0, 5.0, 0.1)



##You can adjust the slope and intercept to verify the changes in the graph

y = 2*(x) + 3

y_noise = 2 * np.random.normal(size=x.size)

ydata = y + y_noise

#plt.figure(figsize=(8,6))

plt.plot(x, ydata,  'bo')

plt.plot(x,y, 'r') 

plt.ylabel('Dependent Variable')

plt.xlabel('Indepdendent Variable')

plt.show()
x = np.arange(-5.0, 5.0, 0.1)



##You can adjust the slope and intercept to verify the changes in the graph

y = 1*(x**3) + 1*(x**2) + 1*x + 3

y_noise = 20 * np.random.normal(size=x.size)

ydata = y + y_noise

plt.plot(x, ydata,  'bo')

plt.plot(x,y, 'r') 

plt.ylabel('Dependent Variable')

plt.xlabel('Indepdendent Variable')

plt.show()
x = np.arange(-5.0, 5.0, 0.1)



##You can adjust the slope and intercept to verify the changes in the graph



y = np.power(x,2)

y_noise = 2 * np.random.normal(size=x.size)

ydata = y + y_noise

plt.plot(x, ydata,  'bo')

plt.plot(x,y, 'r') 

plt.ylabel('Dependent Variable')

plt.xlabel('Indepdendent Variable')

plt.show()
X = np.arange(-5.0, 5.0, 0.1)



##You can adjust the slope and intercept to verify the changes in the graph



Y= np.exp(X)



plt.plot(X,Y) 

plt.ylabel('Dependent Variable')

plt.xlabel('Indepdendent Variable')

plt.show()
X = np.arange(-5.0, 5.0, 0.1)



Y = np.log(X)



plt.plot(X,Y) 

plt.ylabel('Dependent Variable')

plt.xlabel('Indepdendent Variable')

plt.show()
X = np.arange(-5.0, 5.0, 0.1)





Y = 1-4/(1+np.power(3, X-2))





plt.plot(X,Y) 

plt.ylabel('Dependent Variable')

plt.xlabel('Indepdendent Variable')

plt.show()