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
import numpy as np

d=np.mean(doc['x'])

print(d)

t=np.mean(doc['y'])

print(t)
doc['x-x"']=doc['x']-d

doc['y-y"']=doc['y']-t

print(doc)
doc['(x-x")^2']=np.square(doc['x-x"'])

print(doc)
doc['(x-x")X(y-y")']=np.multiply(doc['x-x"'],doc['y-y"'])

print(doc)
import numpy as np

q=np.sum(doc['(x-x")X(y-y")'])

w=np.sum(doc['(x-x")^2'])

print("slope = " )

e=q/w

print(e)
c=t-(e*d)

print(c)

doc['(y-pri)']=(e*doc['x'])+c

print(doc)
doc['(y-pri - y )']=doc['(y-pri)']-doc['y']

doc['(y-pri - y )^2']=np.square(doc['(y-pri - y )'])

print(doc)

g=np.mean(doc['(y-pri - y )^2'])

h=np.sqrt(g)

print(h)