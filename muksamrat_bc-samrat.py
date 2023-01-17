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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os, time, gc
path ="../input"

os.chdir(path)

df = pd.read_csv("../input/crime.csv", encoding = "ISO-8859-1", low_memory=False)
df.head

df.columns
# Distribution of crimes districtwise in descending Order

descending_order = df['DISTRICT'].value_counts().index

sns.countplot("DISTRICT", data = df, order = descending_order)

# District B2 has the highest crime rate followind by C11 and D4 in that order 

# Distribution of crimes by Offense code group top 12
# Distribution of crimes by YEAR

sns.countplot("YEAR", data = df)
# YEARWISE breakup of Crimes by District

sns.catplot(x="DISTRICT",   

            hue="MONTH",      

            col="YEAR",       

            data=df,

            kind="count")
# Distribution of crimes by Offense code group top 15

offense_types = pd.DataFrame({'Count' : df.groupby(["YEAR","OFFENSE_CODE_GROUP"]).size()}).reset_index().sort_values('Count',ascending = False).head(15)

offense_types

sns.barplot(x = "OFFENSE_CODE_GROUP",y= "Count",hue="YEAR", data=offense_types)