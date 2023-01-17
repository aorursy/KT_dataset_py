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
! cat /kaggle/input/fivethirtyeight-marriage-dataset/README.md
data=pd.read_csv("/kaggle/input/fivethirtyeight-marriage-dataset/divorce.csv")

data.shape
man=pd.read_csv("/kaggle/input/fivethirtyeight-marriage-dataset/men.csv")

women=pd.read_csv("/kaggle/input/fivethirtyeight-marriage-dataset/women.csv")

man.shape
man
nopoor1960=man[man["year"]==1960]["nokids_poor_2534"]

nopoor2012=man[man["year"]==2012]["nokids_poor_2534"]

print("Poor and No kids Men \n1960s: ",float(nopoor1960),"\n","2012: ",float(nopoor2012))
nopoor1960=man[man["year"]==1960]["nokids_poor_2534"]

nopoor2012=man[man["year"]==2000]["nokids_poor_2534"]

print("Poor and No kids\n1960s: ",float(nopoor1960),"\n","2000: ",float(nopoor2012))
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.lineplot(data=man,x="year",y="nokids_poor_2534")
nopoor1960=man[man["year"]==1960]["nokids_poor_2534"]

nopoor2012=man[man["year"]==2012]["nokids_poor_2534"]

print("Poor and No kids Women\n1960s: ",float(nopoor1960),"\n","2012: ",float(nopoor2012))
sns.lineplot(data=women,x="year",y="nokids_poor_2534")
sns.relplot(data=women,x="year",y="nokids_poor_2534")
man[["year","nokids_rich_2534"]].corr()
man[["year","nokids_mid_2534"]].corr()
man[["year","nokids_poor_2534"]].corr()
man[["year","kids_rich_2534"]].corr()
man[["year","kids_mid_2534"]].corr()
man[["year","kids_poor_2534"]].corr()
man[["year","GD_3544"]].corr()
women[["year","GD_3544"]].corr()
sns.lineplot(data=women,x="year",y="GD_3544")
sns.relplot(data=women,x="year",y="GD_3544")
man.columns
women[["nokids_all_2534","GD_3544"]].corr()
women[["kids_all_2534","GD_3544"]].corr()
man[["nokids_all_2534","GD_3544"]].corr()
man[["kids_all_2534","GD_3544"]].corr()