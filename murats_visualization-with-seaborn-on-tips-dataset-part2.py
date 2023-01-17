#you can see first kernel (called part1)

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = sns.load_dataset("tips").copy()

x = np.random.normal(size = 100)

x
sns.distplot(x);
sns.distplot(x, kde=False, rug=True);
sns.distplot(x, hist=False,rug = True);
sns.kdeplot(x);
sns.rugplot(x);
df.head(5)
sns.distplot(df.total_bill);
sns.distplot(df.tip, rug=True);
sns.kdeplot(df.tip, shade=True);
sns.kdeplot(x, color="g", shade =True)

sns.kdeplot(df.tip, color = "r", shade =True)

plt.legend();
dfMale = df[df["sex"] == "Male"].copy()
dfFemale = df[df["sex"] == "Female"].copy()
sns.kdeplot(dfMale.total_bill, color = "r", shade =True, alpha =.3)

sns.kdeplot(dfFemale.total_bill, color = "b", shade =True, alpha = .3);
sns.distplot(dfMale.total_bill, color = "r", kde=False)

sns.distplot(dfFemale.total_bill, color = "g", kde=False);
sns.distplot(dfFemale.total_bill, color = "b", kde=False, bins=15);
sns.jointplot(df.tip, df.total_bill, color="r");
sns.pairplot(df);
g = sns.PairGrid(df)

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels = 6);
sns.barplot("size", "total_bill", data = df);
sns.barplot("size", "total_bill", hue = "sex", data = df); 
df.corr()
sns.regplot(x="total_bill", y = "tip", data = df);
sns.lmplot(x="total_bill", y = "tip", data = df);
sns.lmplot("size", "total_bill", data = df);
sns.lmplot(x="size", y="tip", data=df, x_jitter=.2);
sns.lmplot(x="size", y= "tip", data = df, x_estimator = np.mean);
sns.lmplot(x="size", y= "tip", data = df.query("sex == 'Female'"), ci = None,x_estimator = np.mean);
df["big_tip"] = (df.tip / df.total_bill) > .15

df["big_tip"].sample(5)
sns.lmplot("total_bill", "big_tip", data = df, y_jitter = .03);
sns.lmplot("total_bill", "big_tip", data = df, y_jitter = .03, logistic = True);
sns.lmplot("total_bill", "tip", hue = "smoker", data = df);
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=df,

           markers=["o", "x"], palette="Set1");
sns.lmplot(x="total_bill", y="tip", hue="smoker", col="sex", data=df);
sns.lmplot(x = "total_bill", y = "tip", hue = "smoker", col = "sex", row = "time", data = df);
f, ax = plt.subplots(figsize=(8, 7))

sns.regplot(x="total_bill", y="tip", data=df, ax=ax);
sns.lmplot(x = "total_bill", y = "tip", col = "day", data = df, col_wrap = 2, height = 3);
sns.lmplot(x="total_bill", y="tip", col="day", data=df, aspect=.5);
sns.jointplot(x="total_bill", y="tip", data=df, kind="reg");
sns.pairplot(df, x_vars=["total_bill", "size"], y_vars=["tip"], height=5, aspect=.8, kind="reg");
sns.pairplot(df, x_vars=["total_bill", "size"], y_vars=["tip"], hue = "smoker", height=5, aspect=.8, kind = "reg");