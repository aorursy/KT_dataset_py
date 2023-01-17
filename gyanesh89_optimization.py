# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from scipy.optimize import minimize
def objective(x):
    return(-190*x[0]**2-560*x[1])
def con1(x):
    return(-2*x[0]-7*x[1]**2+180)
def con2(x):
    return(-4*x[0]-2*x[1]+300)
def con3(x):
    return (-x[0],-x[1])
x=[1,1]


constraint1={"type":"ineq","fun" :con1}
constraint2={"type":"ineq","fun" :con2}
constraint3={"type":"ineq","fun" :con3}
constraint=(constraint1,constraint2)
minimize(objective,x,method='SLSQP',constraints=constraint)
help(minimize)
