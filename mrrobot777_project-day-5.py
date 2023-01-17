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

ds= pd.read_csv("../input/datasheet.csv")
print(ds)
import numpy as np

a=np.mean(ds["x"])

print(a)

b=np.mean(ds["y"])

print(b)
c=ds["x"]-a

ds["x1"]=c

d=ds["y"]-b

ds["y1"]=d

print(ds)

ds["(x1)^2"]=np.square(ds["x1"])

ds["(x1)^2mean"]=np.mean(ds["(x1)^2"])

print(ds)



ds["(x1)x(y1)"]=np.multiply(ds['x1'],ds['y1'])

print(ds)
e=np.sum(ds["(x1)x(y1)"])

f=np.sum(ds["(x1)^2"])

print("slope=")

g=e/f

print(g)
C=b-g*a

print(C)
ds["C"]=C

ds["y-pri"]=(g*ds["x"])+C

print(ds)

ds["y-pri - y"]=ds["y-pri"]-ds["y"]

ds["(y-pri - y)^2"]=np.square(ds["y-pri - y"])

print(ds)
h=np.mean(ds["(y-pri - y)^2"])

o=np.sqrt(h)

print(o)