# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/statesPopulation.csv', encoding="windows-1252")

data.info()
data.head()
data['State'].value_counts()
data['State'].unique()
len(data['State'].unique())
state_list = list(data['State'].unique())
state_popu = []
for i in state_list :
    x = data[data['State']==i]
    state_populasyon = sum(x.Population)
    state_popu.append(state_populasyon)
data2 = pd.DataFrame({'States':state_list ,'total_population':state_popu})
data2.head()

data['Year'].unique()
data2010= data[data.Year==2010]
state_list_2010 = list(data['State'].unique())
state_popu_2010 = []
for i in state_list_2010 :
    x = data2010[data2010['State']==i]
    state_populasyon_2010 = sum(x.Population)
    state_popu_2010.append(state_populasyon_2010)
data_2010 = pd.DataFrame({'States':state_list_2010 ,'total_population':state_popu_2010})
data_2010.head()
data2014=data[data.Year==2014]
state_list_2014= list(data['State'].unique())
state_popu_2014 = []
for i in state_list_2014:
    x = data2014[data2014['State']==i]
    state_populasyon_2014 = sum(x.Population)
    state_popu_2014.append(state_populasyon_2014)
data_2014 = pd.DataFrame({'States':state_list_2014 ,'total_population':state_popu_2014})
data_2014.head()
s1=data_2014['total_population']
s2=data_2010['total_population']
s3=s1-s2

type(s3)
data3=data_2014.iloc[:,0:1]
dataseries =pd.Series.to_frame(s3)
finishdata =pd.concat([data3,dataseries],axis=1)
finishdata.sort_values(by=['total_population'], ascending=False).head(3)

