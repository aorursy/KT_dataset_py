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
data=pd.read_csv("/kaggle/input/NBA_season1718_salary.csv")

data.head()
data["season17_18"].max()
data["season17_18"].min()
data[data["season17_18"]<20000]
len(data[data["season17_18"]<200*1000])
len(data[data["season17_18"]<200*10000])
len(data[(data["season17_18"]>350*10000) & (data["season17_18"]<1000*10000)])
len(data)
len(data[(data["season17_18"]>350*10000) & (data["season17_18"]<1000*10000)])
data[(data["season17_18"]>2300*10000) ]
maxplayers=data[(data["season17_18"]>2300*10000) ]

otherplayers=data[(data["season17_18"]<=2300*10000) ]
allmaxmoney=maxplayers["season17_18"].sum()

allothermoney=otherplayers["season17_18"].sum()

print("All max money: "+str(allmaxmoney))

print("All other money: "+str(allothermoney))

difference=allmaxmoney-allothermoney

print("Difference: "+str(difference))
teamgrup=data.groupby(data["Tm"])

teampayrolls=teamgrup["season17_18"].sum()

teampayrolls
teamgrup["season17_18"].mean()
teamgrup["season17_18"].count()
teamgrup["season17_18"].count().sort_values()
data[data["Tm"]=="ATL"].sort_values(by=["season17_18"],ascending=False)
data[data["Tm"]=="CLE"].sort_values(by=["season17_18"],ascending=False)
data[data["Tm"]=="POR"].sort_values(by=["season17_18"],ascending=False)
teampayrolls=teamgrup["season17_18"].mean().sort_values()

teampayrolls
teampayrolls=teamgrup["season17_18"].sum().sort_values(ascending=False)

teampayrolls