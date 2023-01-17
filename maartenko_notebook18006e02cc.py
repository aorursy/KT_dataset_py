# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series,DataFrame



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
mountData =pd.read_csv("../input/Mountains.csv")

mountData.head()
mountData.info()
#mountData = mountData.drop(["Parent mountain"], axes=1)

mountData["Ascents bef. 2004"] = mountData["Ascents bef. 2004"].apply(lambda x: pd.to_numeric(x, errors='coerce'))

labels = [ "{0} - {1}".format(i, i + 199) for i in range(7000, 9000, 200) ]

mountData["m_groups"]=pd.cut(mountData["Height (m)"], range(7000, 9200, 200), right=False, labels=labels)

#add extra columns avarage ascents per year after first ascent

#Succes index ascents/ascents+failed ascents

mountData["First ascent"]=mountData["First ascent"].replace("unclimbed","0")

mountData["average ascents per year"]=mountData["Ascents bef. 2004"]/2003-mountData["First ascent"].astype(float)

mountData["succesIndex"]=mountData["Ascents bef. 2004"]/(mountData["Ascents bef. 2004"]+mountData["Failed attempts bef. 2004"])

mountData.plot(x="Height (m)", y="succesIndex")
mountData.plot(x="Height (m)", y="average ascents per year")
# plot

#sns.countplot(x='First ascent', hue='Height (m)', data=mountData, order=[1,0])



#mountData["m_groups"].cat.categories

#mountData["m_groups"].value_counts()

#x=mountData["m_groups"].cat.categories

sns.countplot(y=mountData["m_groups"])