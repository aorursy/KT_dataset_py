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


W = np.array([1,-1,0,0.5]).transpose()

Xi = [np.array([1,-2,0,-1]).transpose(),np.array([0,1.5,0.5,-1]).transpose(), np.array([-1,1,0.5,-1]).transpose()]

D = [-1,-1,1]

b = 0.11

E = 1

iter = 0

m= 0

n = 0





while(E != -0.0):

        net = sum(W.transpose()*Xi[m])

        o = (2/(1 + np.exp(-1*net)))-1

        o_ = ( 0.5 )*(1- (o**2) )

        err = D[m] - o

        print(round(err,1))

        E = round(err,1)

        dw = b * err * o_ * Xi[m]

        W = W + dw

        iter += 1

        m+=1

        if m > 2:

            m = 0

            n += 1

        if E == -0.0:

            break

print("Final Weight Matrix : {}".format(W))

print("Iteration : {}".format(iter))