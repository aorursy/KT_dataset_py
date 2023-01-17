import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df =pd.read_csv("../input/flight-delay-prediction/Jan_2019_ontime.csv")
df.describe()

plt.hist(x=df.DAY_OF_WEEK)
plt.figure(figsize=(20,20))

sns.countplot(x=df.ORIGIN)

sns.distplot(df.DAY_OF_MONTH)
z=df[df.DAY_OF_MONTH==1]

# let see what happened on new year
sns.catplot(x="OP_CARRIER", y="DISTANCE", data=z)
plt.plot(df.groupby("OP_CARRIER")["DISTANCE"].sum())