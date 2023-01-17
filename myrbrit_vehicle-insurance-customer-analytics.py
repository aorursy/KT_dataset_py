# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.





df = pd.read_csv("../input/vehicle-insurance-customer-data/AutoInsurance.csv")
df.shape
df.columns
df.columns = df.columns.str.replace(' ','_')

df.columns
df.head(10)
df.Response.value_counts()
df.Response.value_counts().plot(kind="bar",figsize=(10,7), title="Marketing Response",grid=True)
#Checking the renewal offer column for insights

df.Renew_Offer_Type.value_counts()
byOfferType = df.loc[df.Response == "Yes"].groupby("Renew_Offer_Type")["Customer"].count()/df.groupby("Renew_Offer_Type")["Customer"].count()
ax_byof = byOfferType.plot(kind="bar",grid=True,figsize=(7,7))

ax_byof.set_xlabel("Offer Type")

ax_byof.set_ylabel("Engagement Rate")

plt.show()
byOfferVhClass = df.loc[df.Response == "Yes"].groupby(["Renew_Offer_Type","Vehicle_Class"])["Customer"].count()

byOffrVhPlot = (byOfferVhClass/df.groupby("Renew_Offer_Type")["Customer"].count()).unstack().fillna(0)
ax = byOffrVhPlot.plot(kind="bar",grid=True,figsize=(8,8))

ax.set_xlabel("Offer Type")

ax.set_ylabel("Engagement Rate Of Customer")

plt.show()
bySaleCh = df.loc[df.Response == "Yes"].groupby("Sales_Channel")["Customer"].count()

bySaleChPlot = bySaleCh / df.groupby("Sales_Channel")["Customer"].count()

ax1 = bySaleChPlot.plot(kind="bar",grid=True,figsize=(8,8))

ax.set_xlabel("Sales Channel")

ax.set_ylabel("Engagement Rate Of Customer")

plt.show()