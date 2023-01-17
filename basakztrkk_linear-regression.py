# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# warning library
import warnings
warnings.filterwarnings('ignore')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#import data
data = pd.read_csv("/kaggle/input/linear.csv")



# sklearn library
from sklearn.linear_model import LinearRegression


# linear regression model
lr = LinearRegression()

x = data.metrekare.values.reshape(-1,1)
y = data.fiyat.values.reshape(-1,1)

lr.fit(x, y)  # Verileri x ve y eksenine oturttuk


#y = mx+b

# prediction
b0 = lr.predict([[0]])
print("b0: ",b0)   # Intercept, y eksenini kestigi nokta  b dir. yani y = mx+b 'de x'e sıfır verdiğimizde kalan değer.

b = lr.intercept_
print("b: ",b)   # Intercept, y eksenini kestigi nokta  b dir. yani y = mx+b 'de x'e sıfır verdiğimizde kalan değer.

m = lr.coef_
print("m: ",m)   # Coefficient, egim 


a = np.arange(max(x)).reshape(-1,1)

plt.scatter(x,y) # Gerçek verilerimizi nokta nokta, scatter ile cizdiriyoruz.
plt.plot(a, (m*a+b), color = "red")
plt.xlabel("metrekare")
plt.ylabel("fiyat")
plt.show()




print('Eğim: ', m)
print('Y de kesiştiği yer: ', b)


print("Denklem")
print("y=",m,"x+",b)


