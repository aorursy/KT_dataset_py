# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch
x = 10

a, b = torch.randn(5), torch.randn(5)

#f(x,a ,b )=maxi[(ai−x)**2+bi**2]
def grad(x, a, b):

    a.requires_grad_(True)

    b.requires_grad_(True)

    

    sb = (a-x)**2

    sq = b**2

    sm = sb+sq

    f = sm.max()

     

    f.backward()

    print(a.grad)

    print(b.grad)
grad(x, a, b)