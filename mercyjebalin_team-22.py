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
d=pd.read_csv("/kaggle/input/education-in-india/2015_16_Districtwise.csv")
import matplotlib.pyplot as plt

import seaborn as sns

fig,axes=plt.subplots()

fig.set_size_inches(11.7,8.27)

a=d[d.FEMALE_LIT>64.63583987441132]



vis1=sns.boxplot(data=a,x='STATCD',y='FEMALE_LIT')

d.info()
d.describe()
d.FEMALE_LIT.mean()
d.MALE_LIT.mean()
fig,ax=plt.subplots()

fig.set_size_inches(11.7,8.27)

b=d[d.MALE_LIT>d.MALE_LIT.mean()]



vis2=sns.boxplot(data=b,x='STATNAME',y='MALE_LIT')

vis2.set_xticklabels(vis2.get_xticklabels(), rotation=90)
fig,ax=plt.subplots()

fig.set_size_inches(11.7,8.27)

b=d[d.MALE_LIT<d.MALE_LIT.mean()]



vis2=sns.boxplot(data=b,x='STATNAME',y='MALE_LIT')

vis2.set_xticklabels(vis2.get_xticklabels(), rotation=90)
fig,ax=plt.subplots()

fig.set_size_inches(11.7,8.27)

#vis1=sns.violinplot(data=d,x='STATCD',y='FEMALE_LIT',size=100)

#b=d[d.MALE_LIT<d.MALE_LIT.mean()]

#d['a']=a

#plt.plot(d.a,c='Red',marker='s')

vis3=sns.boxplot(data=d,x='STATNAME',y='MALE_LIT')

vis3.set_xticklabels(vis3.get_xticklabels(), rotation=90)