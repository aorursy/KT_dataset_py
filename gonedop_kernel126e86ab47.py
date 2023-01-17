



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

np.random.seed(180315023)



x = np.linspace(1,10,10)



b0 = np.random.uniform(1,5)

b1 = np.random.uniform(1,5)

b2 = np.random.uniform(1,5)



# To increase the probability

# without enlarging the results

# I will use float numbers 

# Instead of integers



y = b0 + b1*x + b2*x**2
r = np.random.randint(70,130,10)

y = y*r / 100
reg1 = np.polyfit(x,y,1)

reg2 = np.polyfit(x,y,2)



model1 = np.poly1d(reg1)

model2 = np.poly1d(reg2)


print('[Data Points] \n' )

with np.printoptions(precision=2, suppress=True):

    

    print(x)

    print(y)

    print('\n\n[Regression Coefficients] \n\n b0 = (%.2f), b1 = (%.2f), b2 = (%.2f)' %(b0,b1,b2))



print('\n\n[First Order Regression Coefficients]\n\n y(x) = (%.4f)x + (%.4f)' %(reg1[0], reg1[1]))

print('\n r^2 = %.4f' %r2_score(y, model1(x)))



print('\n\n[Second Order Regression Coefficients]\n\n y(x) = (%.4f)x^2 + (%.4f)x + (%.4f)' %(reg2[0], reg2[1], reg2[2]))

print('\n r^2 = %.4f' %r2_score(y, model2(x)))
