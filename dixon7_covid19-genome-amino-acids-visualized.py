# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
tex =open("/kaggle/input/rna-covid19-replacednumbers/9036.txt", "r").read()

tex = tex.replace('\n','')

n=3

tex[0:100]
colr =[tex[i:i+n] for i in range(0, len(tex), n)]

rgbarr = [[int(j) for j in i] for i in colr]



rgbarr.pop()

rgbarr=np.array(rgbarr)

c=int(144)

r = int(len(rgbarr)/c)

flag = np.empty((r,c,3), dtype=np.uint8)

for row in range(r):

     for col in range(c):

            flag[row,col,:] = np.array(rgbarr[c*row + col]*27).astype(float)  

            



plt.figure(figsize = (20,10))

plt.imshow(flag.astype(np.uint8), interpolation='nearest', aspect='auto')

plt.show()