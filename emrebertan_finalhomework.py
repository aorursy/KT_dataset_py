

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


np.random.seed(2402090012)
x=np.linspace(1,10,10)
b0=np.random.randint(1,5)
b1=np.random.randint(1,5)
b2=np.random.randint(1,5)
y=b0+b1*x+b2*x**2
reg1=np.polyfit(x,y,1)
reg2=np.polyfit(x,y,2)
model1=np.poly1d(reg1)
model2=np.poly1d(reg2)
print('Data Points')
with np.printoptions(precision=2,suppress=True):
    print(x)
    print(y)
    print(' Regression Coefficients b0=(%8.3f) b1=(%8.3f) b2=(%8.3f)'%(b0,b1,b2))
print('\nRegression Coefficients for 1st order:')
print(' y(x) = (%8.3f) + (%8.3f) * x'%(reg1[1],reg1[0]))
print(' r2 = %5.3f'%r2_score(y,model1(x)))
print('\nRegression Coefficients for 2nd order:')
print(' y(x) = (%8.3f) + (%8.3f) * x + (%8.3f) * x**2'%(reg2[2],reg2[1],reg2[0]))
print(' r2 = %5.3f'%r2_score(y,model2(x)))
plt.scatter(x,y,c='black')
plt.plot(x,model1(x),'r-')
plt.plot(x,model2(x),'b-')
plt.legend(['1st Order','2nd Order','Data Points'])
plt.grid(True)
plt.show()
np.random.randint(1,574,10)
