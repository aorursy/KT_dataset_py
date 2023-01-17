# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import scipy.optimize as sco 

import scipy.stats as scs

import matplotlib.pyplot as plt

from sklearn import preprocessing
data=pd.read_csv('../input/cnt_km_year_powerPS_minPrice_maxPrice_avgPrice_sdPrice.csv')
data
%matplotlib inline

plt.scatter(data['km'],data['avgPrice'])

plt.xlabel('km')

plt.ylabel('avgPrice')

plt.grid(True)

model=pd.ols(y=data['avgPrice'],x=data['km'])

x=np.arange(0,200000)

plt.plot(model.beta[0]*x+model.beta[1],lw=1.5,color='r')

model
plt.scatter(data['powerPS'],data['avgPrice'])

plt.xlabel('powerPS')

plt.ylabel('avgPrice')

plt.grid(True)

model=pd.ols(y=data['avgPrice'],x=data['powerPS'])

a=np.arange(0,600)

plt.plot(model.beta[0]*a+model.beta[1],lw=2,color='r')

model