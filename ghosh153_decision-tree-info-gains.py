# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/play-tennis/play_tennis.csv')
df
labels=df['play'].unique()
def entropy(data):
    entropy_val=0
    try:
        for i in labels:
            p_label=len(data[data['play']==i])/len(data)
            entropy_val=entropy_val - p_label*math.log2(p_label)
        return entropy_val
    except:
        #retuning 0 because math error occured because log of 0 was to be found, but that means it is also multiplied by 0 so result is 0
        return 0
entropy_parent=entropy(df)
groups=df.groupby('outlook')
categories=df['outlook'].unique()
info_gain=0
for i in categories:
    data=groups.get_group(i)
    entropy_child=entropy(data)
    probability_child=len(data)/len(df)
    info_gain=info_gain-probability_child*entropy_child
print('information gain on outlook column=',entropy_parent+info_gain)
groups=df.groupby('temp')
categories=df['temp'].unique()
info_gain=0
for i in categories:
    data=groups.get_group(i)
    entropy_child=entropy(data)
    probability_child=len(data)/len(df)
    info_gain=info_gain-probability_child*entropy_child
print('information gain on temp column=',entropy_parent+info_gain)
groups=df.groupby('humidity')
categories=df['humidity'].unique()
info_gain=0
for i in categories:
    data=groups.get_group(i)
    entropy_child=entropy(data)
    probability_child=len(data)/len(df)
    info_gain=info_gain-probability_child*entropy_child
print('information gain on humidity column=',entropy_parent+info_gain)
groups=df.groupby('wind')
categories=df['wind'].unique()
info_gain=0
for i in categories:
    data=groups.get_group(i)
    entropy_child=entropy(data)
    probability_child=len(data)/len(df)
    info_gain=info_gain-probability_child*entropy_child
print('information gain on wind column=',entropy_parent+info_gain)
iris=pd.read_csv('/kaggle/input/iris/Iris.csv')
iris
labels=iris['Species'].unique()
labels
def entropy_iris(data):
    entropy_val=0
    try:
        for i in labels:
            p_label=len(data[data['Species']==i])/len(data)
            entropy_val=entropy_val - p_label*math.log2(p_label)
        return entropy_val
    except:
        #retuning 0 because math error occured because log of 0 was to be found, but that means it is also multiplied by 0 so result is 0
        return 0
entropy_parent=entropy(iris)
gain=[]
value=[]
for i in iris['SepalLengthCm']:
    info_gain=0
    d1=iris[iris['SepalLengthCm']<=i]
    d2=iris[iris['SepalLengthCm']>i]
    e1=entropy_iris(d1)
    e2=entropy_iris(d2)
    p1=len(d1)/len(iris)
    p2=len(d2)/len(iris)
    info_gain=entropy_parent - e1*p1 -e2*p2
    gain.append(info_gain)
    value.append(i)
max_gain=max(gain)
print('information gain on SepalLengthCm colum =',max_gain)
print('Value to split node at =', value[np.where(np.array(gain)==max_gain)[0][0]])
gain=[]
value=[]
for i in iris['SepalWidthCm']:
    info_gain=0
    d1=iris[iris['SepalWidthCm']<=i]
    d2=iris[iris['SepalWidthCm']>i]
    e1=entropy_iris(d1)
    e2=entropy_iris(d2)
    p1=len(d1)/len(iris)
    p2=len(d2)/len(iris)
    info_gain=entropy_parent - e1*p1 -e2*p2
    gain.append(info_gain)
    value.append(i)
max_gain=max(gain)
print('information gain on SepalWidthCm colum =',max_gain)
print('Value to split node at =', value[np.where(np.array(gain)==max_gain)[0][0]])
gain=[]
value=[]
for i in iris['PetalLengthCm']:
    info_gain=0
    d1=iris[iris['PetalLengthCm']<=i]
    d2=iris[iris['PetalLengthCm']>i]
    e1=entropy_iris(d1)
    e2=entropy_iris(d2)
    p1=len(d1)/len(iris)
    p2=len(d2)/len(iris)
    info_gain=entropy_parent - e1*p1 -e2*p2
    gain.append(info_gain)
    value.append(i)
max_gain=max(gain)
print('information gain on PetalLengthCm colum =',max_gain)
print('Value to split node at =', value[np.where(np.array(gain)==max_gain)[0][0]])
gain=[]
value=[]
for i in iris['PetalWidthCm']:
    info_gain=0
    d1=iris[iris['PetalWidthCm']<=i]
    d2=iris[iris['PetalWidthCm']>i]
    e1=entropy_iris(d1)
    e2=entropy_iris(d2)
    p1=len(d1)/len(iris)
    p2=len(d2)/len(iris)
    info_gain=entropy_parent - e1*p1 -e2*p2
    gain.append(info_gain)
    value.append(i)
max_gain=max(gain)
print('information gain on PetalWidthCm colum =',max_gain)
print('Value to split node at =', value[np.where(np.array(gain)==max_gain)[0][0]])
