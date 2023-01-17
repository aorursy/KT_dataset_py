# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
df = pd.read_csv('../input/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv')
df.shape
df.columns
df.columns = df.columns.str.replace(' ','_')
df.columns
df.head()
df.dtypes
df.dtypes.groupby(df.dtypes.values).count()
df.Response.value_counts()
df.Response.value_counts().plot(kind="bar",figsize=(10,7), title="Marketing Response",grid=True);
(df.Response.value_counts()/df.shape[0])*100

df.Renew_Offer_Type.value_counts()
by_offer_type = df.loc[df.Response == "Yes"].groupby("Renew_Offer_Type")["Customer"].count()/df.groupby("Renew_Offer_Type")["Customer"].count()
ax_byof = by_offer_type.plot(kind="bar",grid=True,figsize=(7,7))

ax_byof.set_xlabel("Offer Type")

ax_byof.set_ylabel("Engagement Rate")

plt.show()
df.Vehicle_Class.value_counts()
by_offer_vh_class = df.loc[df.Response == "Yes"].groupby(["Renew_Offer_Type","Vehicle_Class"])["Customer"].count()
by_of_vh_plot = (by_offer_vh_class/df.groupby("Renew_Offer_Type")["Customer"].count()).unstack().fillna(0)
by_of_vh_plot
ax = by_of_vh_plot.plot(kind="bar",grid=True,figsize=(8,8))

ax.set_xlabel("Offer Type")

ax.set_ylabel("Engagement Rate Of Customer")

plt.show()
by_sale_ch = df.loc[df.Response == "Yes"].groupby("Sales_Channel")["Customer"].count()

by_sale_ch
by_sale_ch_plot = by_sale_ch / df.groupby("Sales_Channel")["Customer"].count()

ax1 = by_sale_ch_plot.plot(kind="bar",grid=True,figsize=(8,8))

ax.set_xlabel("Sales Channel")

ax.set_ylabel("Engagement Rate Of Customer")

plt.show()
df.Months_Since_Policy_Inception.describe()
bins = [0,25,50,75,100]

grp = ["0-25","25-50","50-75","75-100"]

# set include_lowest or you will get missing values

df["Inception_cat"] = pd.cut(df["Months_Since_Policy_Inception"],bins,labels= grp,include_lowest=True)
df.Inception_cat.value_counts()
by_inc_type = df.loc[df.Response == "Yes"].groupby("Inception_cat")["Customer"].count()/df.groupby("Inception_cat")["Customer"].count()
ax = by_inc_type.plot(kind="bar",grid=True,figsize=(8,8))

ax.set_xlabel("Months Since Inception")

ax.set_ylabel("Engagement Rate Of Customer")

plt.show()