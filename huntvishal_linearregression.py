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

doc=pd.DataFrame({'x':[1,2,4,3,5],'y':[1,3,3,2,5]})

print(doc)
import numpy as np

s=np.mean(doc['x'])

print(s)

t=np.mean(doc['y'])

print(t)
g=doc['x']-s

doc['x-x"']=g

h=doc['y']-t

doc['y-y"']=h

print(doc)
doc['(x-x")^2']=np.square(g)

print(doc)
doc['(x-x")(y-y")']=np.multiply(g,h)

print(doc)
q=np.sum(doc['(x-x")(y-y")'])

w=np.sum(doc['(x-x")^2'])

e=q/w

print("slope=",e)
c=t- s*e

print(c)
a=(e*doc['x']) + c

doc['y"']=a

print(doc)
doc['y-y""']=a-doc['y']

doc['(y-y"")^2']=np.square(doc['y-y""'])

print(doc)
o=np.mean(doc['(y-y"")^2'])

j=np.sqrt(o)

print(j)