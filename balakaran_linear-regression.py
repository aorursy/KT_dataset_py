import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline
# generate 3 independent variables x1,x2,x3 randomly drawn from standardised normal 

# distributions but with positive correlations with the dependent variable y1.

# The dependent variable is also randomly drawn from a standardised normal.



# We also enforced a strong correlation between two independent variables x2, x3



import numpy.random as nr

nr.seed()

y = nr.normal(0,1, (10000) )

def correlatedValue (x, r):

  r2 = r**2

  ve = 1-r2

  SD = np.sqrt(ve)

  e  = nr.normal(0, SD, (x.size))

  y  = r*x + e

  return(y)

x1 = correlatedValue(y, r=0.6)

x2 = correlatedValue(y, r=0.8)

x3 = correlatedValue(x2, r=0.9)

corx1y = np.corrcoef(y,x1) 

corx2y = np.corrcoef(y,x2)

corx3y = np.corrcoef(y,x3) 

corx2x3 = np.corrcoef(x2,x3)

print ('cor (x1,y) = ', corx1y[0,1])

print ('cor (x2,y) = ', corx2y[0,1])

print ('cor (x3,y) = ', corx3y[0,1])

print ('cor (x2,x3) =', corx2x3[0,1])
x=np.array([x1,x2,x3])

for i in [1,2,3] :

    plt.subplot (2,2,i)

    plt.scatter (x[i-1],y)

    plt.xlabel ('x'+ str (i) + ' vs y')

plt.subplot (2,2,4)    

plt.scatter (x[1],x[2])

plt.xlabel ('x2 vs x3')

plt.show()
#OLS implementation

from scipy import stats

import numpy as np

iv = x2

dv = y



slope, intercept, cor, p_value, std_err = stats.linregress(iv,dv)

# To get coefficient of determination (r_squared)

print ('correlation= ', cor, 'r-squared= ', cor**2, 'slope= ', slope, 'intercept=', intercept )
import tables

import numpy

ivariable = x2

dvariable = y

hdf5_path = "mydata.hdf5"

hdf5_file = tables.open_file(hdf5_path, mode='w')

iv_storage = hdf5_file.create_array(hdf5_file.root, 'iv', ivariable)

dv_storage = hdf5_file.create_array(hdf5_file.root, 'dv', dvariable)

hdf5_file.close()



read_hdf5_file = tables.open_file(hdf5_path, mode='r')

hdf5_ivariable = read_hdf5_file.root.iv[:]

hdf5_dvariable = read_hdf5_file.root.dv[:]

read_hdf5_file.close()



slope, intercept, cor, p_value, std_err = stats.linregress(hdf5_ivariable, hdf5_dvariable)

print ('correlation= ', cor, 'r-squared= ', cor**2, 'slope=', slope, 'intercept=', intercept)