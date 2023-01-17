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
train_data=pd.read_csv('/kaggle/input/train_data.csv')
test_data=pd.read_csv('/kaggle/input/test_data.csv')
features = ['Gender', 'Height', 'Weight', 'Index']
# Training Data
traing = train_data.loc['0':, features[0]].values
trainh = train_data.loc['0':, features[1]].values 
trainw = train_data.loc['0':, features[2]].values 

# Testing Data
testg = test_data.loc['0':, features[0]].values
testh = test_data.loc['0':, features[1]].values 
testw = test_data.loc['0':, features[2]].values
K=7
test=0
ts = traing.size
tst = testg.size

maximums=[]
finalg=[]
males=0
females=0
eqd=[]
for j in range(0,ts):
    dist=((float(testh[test])-float(trainh[j]))**2+(float(testw[test])-float(trainw[j]))**2)**.5
    eqd.append(dist)
eqd.sort()

for q in range(0,K):
    max1 = eqd[ts-q-1]
    maximums.append(max1)
    for w in range(0,ts):
        if max1 == eqd[w]:
            predictg=traing[w]

    finalg.append(predictg)
    if finalg[q]=="Female":
        females=females+1
    elif finalg[q]=="Male":
        males=males+1

print('K was taken ',K)
print("Maximum values' array = ", maximums)
print("Genders Array", finalg)
print("No. of Males = ", males)
print("No. of Females = ", females)

if females>males:
    print("This is Female")
else:
    print("This is Male")