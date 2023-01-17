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
import pandas as pd

doc = pd.read_csv("../input/doc.csv")
print(doc)
xm=doc['x'].mean()

print(xm)

ym=doc['y'].mean()

print(ym)
doc['x-xm']=doc['x']-xm

doc['y-ym']=doc['y']-ym

doc['mul']=doc['x-xm']*doc['y-ym']

doc['sqx']=doc['x-xm']*doc['x-xm']

print(doc)
sum_mul=np.sum(doc['mul'])

print(sum_mul)

sum_sqx=np.sum(doc['sqx'])

print(sum_sqx)
b1=sum_mul/sum_sqx

print(b1)

b0=ym-b1*xm

print(b0)
doc['piy']=b0+b1*doc['x']

doc['piy-yi']=doc['piy']-doc['y']

doc['piy-yi^sq']=doc['piy-yi']*doc['piy-yi']

print(doc)
rhse=np.sqrt(np.sum(doc['piy-yi^sq'])/len(doc))

print(rhse)