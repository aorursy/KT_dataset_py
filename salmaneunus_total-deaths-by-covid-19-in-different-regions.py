import pandas as pd

d = pd.read_csv("../input/total-death-by-covid19-in-different-regions/total-covid-deaths-region.csv")
d.describe()
d.head()
d.tail()
print(d.iloc[500,0:])
d.Date.dtype
max(d.iloc[0:,3])
max(d.Entity)
d
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

sns.lineplot(x = d.Date,y =d.iloc[0:,3],hue = d.Entity)

plt.xlabel("From December 2019 to June 2020")
plt.figure(figsize=(20,10))

sns.boxplot(x = d.Date,y = d.iloc[0:,3])
plt.figure(figsize=(20,20))

sns.barplot(x = d.Date,y =d.iloc[600:800,3])

plt.xlabel("From December 2019 to June 2020")
plt.figure(figsize=(20,20))

sns.barplot(x = d.Date,y =d.iloc[1200:2000,3])

plt.xlabel("From December 2019 to June 2020")
plt.figure(figsize=(20,20))

sns.scatterplot(data =d.iloc[600:,3])

plt.xlabel("From December 2019 to June 2020")

plt.ylabel("Total deaths in different regions")
plt.figure(figsize=(20,20))

sns.swarmplot(x = d.Date,y =d.iloc[12000:20000,3])

plt.xlabel("From December 2019 to June 2020")