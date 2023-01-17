# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as ns

import seaborn as sns

import matplotlib.pyplot as plt

import time

import os

from collections import Counter
# Tells us about the water level in all the available reservoir for the last 15 year

data_l= pd.read_csv("../input/chennai_reservoir_levels.csv",index_col=["Date"])

# Denotes the amount of rainfall over the same period inn  the region of all the mentioned reservoirs

data_r= pd.read_csv("../input/chennai_reservoir_rainfall.csv", index_col=["Date"])
data_l.index = pd.to_datetime(data_l.index)

data_r.index = pd.to_datetime(data_r.index)
data_l.head()
data_r.head()
f, ((ax1,ax2),(ax3,ax4)) =plt.subplots(2,2, figsize=(25,10))

data_l["POONDI"].resample("Y").mean().plot(kind= "line",ax= ax1, title= "POONDI water level over years", color="r")

data_l["CHOLAVARAM"].resample("Y").mean().plot(kind= "line",ax= ax2, title= "CHOLAVARAM water level over years", color= "y")

data_l["REDHILLS"].resample("Y").mean().plot(kind= "line",ax= ax3, title= "REDHILLS water level over years",color= "g")

data_l["CHEMBARAMBAKKAM"].resample("Y").mean().plot(kind= "line",ax= ax4, title= "CHEMBARAMBAKKAM water level over years", color= "b")
f, ((ax1,ax2),(ax3,ax4)) =plt.subplots(2,2, figsize=(25,10))

data_l[data_l.index.year==2010]["POONDI"].resample("M").sum().plot(kind= "line",ax= ax1, title= "POONDI water level over month for 2010", color="r")

data_l[data_l.index.year==2010]["CHOLAVARAM"].resample("M").sum().plot(kind= "line",ax= ax2, title= "CHOLAVARAM water level over month for 2010", color="b")

data_l[data_l.index.year==2010]["REDHILLS"].resample("M").sum().plot(kind= "line",ax= ax3, title= "REDHILLS water level over month for 2010", color="y")

data_l[data_l.index.year==2010]["CHEMBARAMBAKKAM"].resample("M").sum().plot(kind= "line",ax= ax4, title= "CHEMBARAMBAKKAM water level over month for 2010", color="g")
f, ((ax1,ax2),(ax3,ax4)) =plt.subplots(2,2, figsize=(25,10))

data_l[data_l.index.year==2011]["POONDI"].resample("M").sum().plot(kind= "line",ax= ax1, title= "POONDI water level over month for 2011", color="r")

data_l[data_l.index.year==2011]["CHOLAVARAM"].resample("M").sum().plot(kind= "line",ax= ax2, title= "CHOLAVARAM water level over month for 2011", color="g")

data_l[data_l.index.year==2011]["REDHILLS"].resample("M").sum().plot(kind= "line",ax= ax3, title= "REDHILLS water level over month for 2011", color="b")

data_l[data_l.index.year==2011]["CHEMBARAMBAKKAM"].resample("M").sum().plot(kind= "line",ax= ax4, title= "CHEMBARAMBAKKAM water level over month for 2011", color="y")
f, ((ax1,ax2),(ax3,ax4)) =plt.subplots(2,2, figsize=(25,10))

data_l[data_l.index.year==2012]["POONDI"].resample("M").sum().plot(kind= "line",ax= ax1, title= "POONDI water level over month for 2012", color="y")

data_l[data_l.index.year==2012]["CHOLAVARAM"].resample("M").sum().plot(kind= "line",ax= ax2, title= "CHOLAVARAM water level over month for 2012", color="r")

data_l[data_l.index.year==2012]["REDHILLS"].resample("M").sum().plot(kind= "line",ax= ax3, title= "REDHILLS water level over month for 2012", color="b")

data_l[data_l.index.year==2012]["CHEMBARAMBAKKAM"].resample("M").sum().plot(kind= "line",ax= ax4, title= "CHEMBARAMBAKKAM water level over month for 2012", color="g")
f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(25,14))

sns.distplot(data_l["POONDI"],ax=ax1)

sns.distplot(data_l["CHOLAVARAM"],ax=ax2)

sns.distplot(data_l["REDHILLS"],ax=ax3, color="r")

sns.distplot(data_l["CHEMBARAMBAKKAM"],ax=ax4)
f,ax =plt.subplots(1,1,figsize=(25,11))

data_l[data_l.index.year==2010].resample("M").max().plot(kind="bar",ax =ax)
data_r.head(10)
f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(20,10))

data_r["POONDI"].resample("M").mean().plot(kind="line",ax=ax1,color="b")

data_r["CHOLAVARAM"].resample("M").mean().plot(kind="line",ax=ax2,color="r")

data_r["REDHILLS"].resample("M").mean().plot(kind="line",ax=ax3,color="g")

data_r["CHEMBARAMBAKKAM"].resample("M").mean().plot(kind="line",ax=ax4,color="g")
f,((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2, figsize=(20,10))

data_r[data_r.index.year==2015]["POONDI"].resample("M").mean().plot(kind="line",ax=ax1,color="r",title="POONDI 2015 RAINFALL")

data_l[data_l.index.year==2015]["POONDI"].resample("M").mean().plot(kind="line",ax=ax2,color="b", title="POONDI 2015 RESERVOIR LEVEL")

f.tight_layout()

data_r[data_r.index.year==2015]["CHOLAVARAM"].resample("M").mean().plot(kind="line",ax=ax3,color="r",title="CHOLAVARAM 2015 RAINFALL")

data_l[data_l.index.year==2015]["CHOLAVARAM"].resample("M").mean().plot(kind="line",ax=ax4,color="b",title="CHOLAVARAM 2015 RESERVOIR LEVEL")

f.tight_layout()

data_r[data_r.index.year==2015]["REDHILLS"].resample("M").mean().plot(kind="line",ax=ax5,color="r",title="REDHILLS 2015 RAINFALL")

data_l[data_r.index.year==2015]["REDHILLS"].resample("M").mean().plot(kind="line",ax=ax6,color="b",title="REDHILLS 2015 RESERVOIR LEVEL")

f.tight_layout()

data_r[data_r.index.year==2015]["CHEMBARAMBAKKAM"].resample("M").mean().plot(kind="line",ax=ax7,color="r",title="CHEMBARAMBAKKAM 2015 RAINFALL")

data_l[data_l.index.year==2015]["CHEMBARAMBAKKAM"].resample("M").mean().plot(kind="line",ax=ax8,color="b", title="CHEMBARAMBAKKAM 2015 RESERVOIR LEVEL")
#sns.swarmplot(data_l["POONDI"].resample("M").mean(),color="r")

f,((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2, figsize=(20,10))

sns.swarmplot(data_l["POONDI"].resample("M").mean(),color="r",ax=ax1)

sns.swarmplot(data_r["POONDI"].resample("M").mean(),color="b",ax=ax2)

sns.swarmplot(data_l["CHOLAVARAM"].resample("M").mean(),color="r",ax=ax3)

sns.swarmplot(data_r["CHOLAVARAM"].resample("M").mean(),color="b",ax=ax4)

sns.swarmplot(data_l["REDHILLS"].resample("M").mean(),color="r",ax=ax5)

sns.swarmplot(data_r["REDHILLS"].resample("M").mean(),color="b",ax=ax6)

sns.swarmplot(data_l["CHEMBARAMBAKKAM"].resample("M").mean(),color="r",ax=ax7)

sns.swarmplot(data_r["CHEMBARAMBAKKAM"].resample("M").mean(),color="b",ax=ax8)