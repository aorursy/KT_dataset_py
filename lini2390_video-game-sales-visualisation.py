import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
os.chdir("../input")

os.listdir()
data = pd.read_csv("vgsales.csv")
data.shape
data.columns
data.isnull().values.any()
data.head()
data.tail()


dt = data[['NA_Sales',

       'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].sum().reset_index()

dt.columns = ['Area','Sales_tot']
dt
ax = sns.barplot(x="Area", y="Sales_tot", data=dt)

ax.set_title("Bar plot total sales vs area")
plt.pie(dt['Sales_tot'],labels=dt['Area'])
dtg = data.groupby(['Genre'])

dtg_t = dtg['NA_Sales',

       'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'].aggregate(np.sum)
dtg_t
dtg_t.plot()
dtm = data.groupby(['Year'])

dtm_t = dtm['NA_Sales',

       'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'].aggregate(np.mean)

dtm_t
dtm_t.plot()