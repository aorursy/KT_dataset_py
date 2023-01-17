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

from scipy.optimize import linprog

from numpy.linalg import solve
f = -1*np.array([5,1]); #Objective function

                       #maximize 5*x1 + x2 equivalent to minimize -5*x1 - x2 

 
#Inequality constraints                       

A_ineq = np.array([[-1,2],[1,-1]]);

b_ineq = np.array([3,2]);



lb_ub = (0,None); #Bounds on design variables



res = linprog(f, A_ub=A_ineq, b_ub=b_ineq,bounds=lb_ub);
print('Value of objective function at optimal solution = ', -res.fun);

print(' Solution x = ', res.x);