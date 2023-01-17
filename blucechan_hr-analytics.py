import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



df = pd.read_csv("../input/HR_comma_sep.csv")

df.head(2)
df.describe()
df.info()
df["Work_accident"].value_counts()
df["sales"].value_counts()
#assuming monthly workdays = 20

df["average_daily_hours"] = df["average_montly_hours"]/20



df.head(2)
document1 = []

for hours in df["average_daily_hours"]:

    if hours >= 12.00:

        document1.append("12-15")

    elif (hours <= 12.00 and hours >=7.95):

        document1.append("8-12")

    else:

        document1.append("4-7")



df["average_daily_hours"] = document1
sns.countplot(df.number_project)
sns.countplot(x="average_daily_hours",data=df)
sns.countplot(x='sales',data=df)
plt.figure(figsize=(12,10))

sns.heatmap(df.corr(),annot=True)