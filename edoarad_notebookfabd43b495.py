# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")
train.values.max()
for i in range(30,2,-1):

    if 784/i == int(784/i):

        print(i)

        break
784/28
fit=[0]*25

for l in range(15,40):

    i = l-15

    for h in range(783-40):

        s = "pixel"+str(h)

        sl= "pixel"+str(h+l)

        v = train[s].values - train[sl].values

        v = list(map(lambda x: x*x, v))

        fit[i] += sum(v)

print(fit)
fit.index(min(fit))+15
a