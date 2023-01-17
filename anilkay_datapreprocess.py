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
data = pd.read_csv("../input/who_suicide_statistics.csv")
#İ will back after my regular worklods
data.head()
print(type(data))
turkeydata=data[(data.country=='Turkey')]
len(turkeydata) #TURKEY has only 84 record
femalecount=len(turkeydata[(turkeydata.sex=='female')])
malecount=len(turkeydata[(turkeydata.sex=='male')])
print("Turkish  female death: "+str(femalecount)+" male: death: "+str(malecount))
#Turks is very strict for their data public.
germany=data[data.country=='Germany']

len(germany)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["sex"]=le.fit_transform(data["sex"])
#Female is 0 and male is 1

data.head()

data.tail()



dataonehot






germany















turkeydata




