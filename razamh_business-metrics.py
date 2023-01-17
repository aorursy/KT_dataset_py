# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/businesss/nps.csv")

df.head()
df.info()
df["event_date"] = pd.to_datetime(df["event_date"])
df.info()
df["score"].unique()
sorted(df["score"].unique())
year = df["event_date"].dt.year

month = df["event_date"].dt.month

yearmonth = year*100 + month

yearmonth.tail()
df["yearmonth"] = yearmonth
df.head()
def category(score):

    if score in range(0,7):

        return "Detractor"

    elif score in (7,8):

        return "Passive"

    elif score in (9,10):

        return "Promoter"
df["category"] = df["score"].apply(category)
df.head()
nps = df.pivot_table(index = "yearmonth",columns =  "category",aggfunc  = "size")

nps.head()
nps["total_responses"] = nps.sum(axis = 1)
nps.head()
nps["nps"] = (nps["Promoter"] - nps["Detractor"])/nps["total_responses"]

nps["nps"] = (nps["nps"]*100).astype(int)
nps.head()
# Visulaization view

ax = nps.reset_index().plot("yearmonth", "nps", kind = "line", legend = False,

                          figsize = (12,6))

ax.set_xticks(nps.index)

ax.set_xticklabels(nps.index , rotation = 45)

plt.xlabel("")

plt.show()
subs = pd.read_csv("/kaggle/input/businesss/muscle_labs.csv", parse_dates = ["end_date", "start_date"])

subs.head()
subs.info()
subs["churn_month"] = subs["end_date"].dt.year*100 + subs["end_date"].dt.month
subs.head()
monthly_churn = pd.DataFrame({"total_churn":subs.groupby("churn_month").size()})
monthly_churn.head()
monthly_churn.index.name = None
monthly_churn.head()
years = list(range(2011,2016))

months = list(range(1,13))

yearmonths = [y*100+m for y in years for m in months]
yearmonths
churn = pd.DataFrame({"yearmonth":yearmonths})

churn.head()
churn = pd.merge(churn, monthly_churn, how = "left", left_on = "yearmonth", right_index = True)



churn.head()
churn.fillna(0, inplace = True)
churn.head()
churn["total_churn"] = churn["total_churn"].astype(int)
def get_customer(yearmonth):

    import datetime as dt

    year = yearmonth//100

    month = yearmonth-year*100

    date = dt.datetime(year,month,1)

    return ((subs['start_date']<date) & (date <= subs['end_date'])).sum()
churn['total_customer'] = churn['yearmonth'].apply(get_customer) 
churn.head()
churn['churn_rate'] = churn['total_churn']/churn['total_customer']
churn.info()
churn.head()
churn.dropna(axis = 0 , inplace = True)
churn.head()
churn = churn[churn['yearmonth']<201422]
churn['yearmonth'] = churn['yearmonth'].astype(str)
from matplotlib.patches import Ellipse
ax = churn.plot.line(x = "yearmonth" , y = "churn_rate" , figsize = (12,6), rot = 45, marker = ".")



start , end = ax.get_xlim()

ax.set_xticks(np.arange(2,end,3))

ax.set_xticklabels(churn['yearmonth'][1::3])

x = 35

y = churn.loc[churn['yearmonth'] == '201312' , "churn_rate"].iloc[0]

circle = Ellipse((x,y),5 , 0.05 , color = "sandybrown", fill = False)

ax.add_artist(circle)

ax.set_label("")

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.tick_params(left = False , bottom = False)

plt.legend("")

plt.show()