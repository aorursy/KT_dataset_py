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
w1=np.array([1,0]).transpose()

w2=np.array([0,-1]).transpose()

xi=[np.array([0.8,0.6]).transpose(),np.array([0.17,-0.98]).transpose(),np.array([0.707,0.707]).transpose(),np.array([0.34,-0.93]),

    np.array([0.6,0.8]).transpose()]

b=2

w=np.array([0])

for i in range(len(xi)):

    net1=sum(w1.transpose()*xi[i])

    net2=sum(w2.transpose()*xi[i])

    if net1>net2:

         dw=b*(xi[i]-w1)

         print()

    else:

        dw=b*(xi[i]-w2)

    w=w+dw

print("final weight:",w)