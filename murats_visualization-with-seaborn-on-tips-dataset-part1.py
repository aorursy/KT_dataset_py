import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
tips = sns.load_dataset("tips")

df = tips.copy()

df.sample(7)
df.info()
df.describe().T
df.isnull().sum()
df.corr()
sns.set(style="darkgrid")

sns.relplot(x = "total_bill", y = "tip", data = df);
sns.relplot(x = "total_bill", y = "tip", hue= "smoker", data = df);
sns.scatterplot(x = "total_bill", y = "tip", hue= "sex", data = df);
sns.scatterplot(x = "total_bill", y = "tip", hue= "smoker", style= "smoker", data = df);
sns.relplot(x = "total_bill",

            y = "tip",

            hue = "smoker",

            style = "time",

            height = 6,

            data = tips);
sns.relplot(x = "total_bill", y = "tip", hue = "size", height = 7, data = df);
sns.relplot(x = "total_bill", y = "tip", size = "size", sizes = (20,100), hue = "size", data = df);
sns.relplot(x = "total_bill", y = "tip", hue = "smoker", col = "time", data = df);
sns.relplot(x = "total_bill", y = "tip", hue = "smoker", col = "sex", data = df);
sns.relplot(x = "total_bill", y = "tip", hue = "smoker", col = "day", data = df);
sns.catplot(x = "day", y = "total_bill", data = df);
sns.catplot(x = "day", y = "total_bill", hue = "sex", data = df);
sns.catplot(x = "day", y = "total_bill", jitter = False, hue = "sex", alpha = .33, data = df);
sns.swarmplot(x = "day", y = "total_bill", data = df);
sns.swarmplot(x = "day", y = "total_bill", hue = "sex", alpha = .75, data = df);
sns.swarmplot(x ="size", y = "total_bill", data = df);
sns.swarmplot(x ="size", y = "total_bill", hue = "sex", alpha =.7, data = df);
sns.catplot(x = "smoker", y = "tip", order = ["No", "Yes"], data = df);
sns.catplot(x = "day", y = "total_bill", hue = "time", alpha = .5, data = df);
sns.swarmplot(x = "day", y = "total_bill", hue = "time", alpha = .5, data = df);
sns.boxplot(x = "day", y = "total_bill", data = df);
sns.boxplot(x = "day", y = "total_bill", hue = "sex", data = df);
sns.boxplot(x = "day", y = "total_bill", hue = "smoker", data = df);
df["weekend"] = df["day"].isin(["Sat","Sun"])

df.sample(5)
sns.boxplot(x = "day", y = "total_bill", hue = "weekend", data = df);
sns.boxenplot(x= "sex", y = "tip", hue = "smoker", data = df);
sns.violinplot(x ="day", y = "total_bill", hue = "time", data = df);
sns.violinplot(x ="day", y = "total_bill", hue = "time", bw = .15, data = df);
sns.violinplot(x ="day", y = "total_bill", hue = "smoker", bw = .25, split = True, data = df);
sns.violinplot(x="day", y="total_bill", hue="smoker", bw=.25, split=True, palette= "pastel", inner= "stick", data=df);
sns.violinplot(x = "day", y = "total_bill", inner = None, data = df)

sns.swarmplot(x = "day", y = "total_bill", color = "k", size = 3, data = df);
sns.barplot(x = "sex", y= "total_bill", hue = "smoker", data = df);
sns.barplot(x = "day", y= "tip", hue = "smoker", palette = "ch:.25", data = df);
sns.countplot(x = "day", hue ="sex", data = df);
sns.countplot(x = "sex", hue = "smoker", palette = "ch:.25", data = df);
sns.countplot(x = "day", hue = "size", palette = "ch:.25", data = df);
sns.pointplot(x = "day", y = "tip", data= df);
sns.pointplot(x = "day", y = "size", hue = "sex", linestyles = ["-", "--"], data= df);
f, ax = plt.subplots(figsize = (7,3))

sns.countplot(x = "day", hue= "smoker", data = df);
sns.catplot(x="day", y = "total_bill", hue = "smoker", col = "time", data = df);
sns.catplot(x = "day", y = "total_bill", col = "sex", kind="box", data = df);
# to be continued.