# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

label = train["label"]
digit = []
train = train.drop('label',axis=1)

def euclid(a,b):
    a = np.array(a, dtype="float")
    b = np.array(b, dtype="float")
    return np.sum(np.power(np.subtract(a,b),2))
ans = []
val=False
for i in test.iterrows():
    if not val:
        val=True
        continue
    count = {}
    l=-1
    dist = []
    for j in train.iterrows():
        if l==-1: 
            l+=1
            continue
        dist.append((label[l],euclid(i,j)))
        l+=1
    dist.sort(key=lambda x: x[1])[:20]
    for i,j in dist:
        count[i]=count.get(i,0)+1
    count = sorted(count, key=lambda k: count[k])
    ans.append(count[0])
print (ans[:10])
# Any results you write to the current directory are saved as output.