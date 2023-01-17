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
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
os.getcwd()

crime = pd.read_csv("../input/crime.csv", encoding = "ISO-8859-1")

crime.head

crime.columns
crime.head()
crime.describe()
#Distribution of crimes according to months

import seaborn as sns

sns.distplot(crime['MONTH'],kde=False,bins=30)
#District-wise crime rate

sns.countplot(x='DISTRICT',data=crime)
#UCR-part wise crime rate

sns.countplot(x='UCR_PART',data=crime)
#correlation between columns

df=crime.corr()

sns.heatmap(df,annot=True)
# Distribution of offense code group with UCR part

crimes_offense = pd.DataFrame({'Count' : crime.groupby(["UCR_PART","OFFENSE_CODE_GROUP"]).size()}).reset_index().sort_values('Count',ascending = False).head(10)

crimes_offense

sns.barplot(y = "OFFENSE_CODE_GROUP",x= "Count",hue="UCR_PART", data=crimes_offense)

plt.xticks(rotation=75)