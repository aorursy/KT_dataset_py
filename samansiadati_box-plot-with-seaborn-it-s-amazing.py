import seaborn as sns
tips = sns.load_dataset("tips")
tips
tips.total_bill.value_counts()
tips.total_bill.describe()
ax = sns.boxplot(x=tips["total_bill"])
ax = sns.boxplot(x="day", y="total_bill", data=tips)
ax = sns.boxplot(x="day", y="total_bill", hue="smoker",
                 data=tips)
ax = sns.boxplot(x="day", y="total_bill", hue="smoker",
                 data=tips, palette="Set3")
ax = sns.boxplot(x="day", y="total_bill", hue="time",
                 data=tips)
ax = sns.boxplot(x="day", y="total_bill", hue="time",
                 data=tips, linewidth=2.5)
ax = sns.boxplot(x="time", y="tip", data=tips)
ax = sns.boxplot(x="time", y="tip", data=tips,
                 order=["Dinner", "Lunch"])
tips["weekend"] = tips["day"].isin(["Sat", "Sun"])
tips["weekend"]
ax = sns.boxplot(x="day", y="total_bill", hue="weekend",
                 data=tips)
ax = sns.boxplot(x="day", y="total_bill", hue="weekend",
                 data=tips, dodge=False)
ax = sns.swarmplot(x="day", y="total_bill", data=tips)
ax = sns.swarmplot(x="day", y="total_bill", data=tips, color=".25")
ax = sns.boxplot(x="day", y="total_bill", data=tips)
ax = sns.swarmplot(x="day", y="total_bill", data=tips, color=".25")
g = sns.boxplot(x="sex", y="total_bill",hue="smoker", data=tips)
g = sns.boxplot(x="time", y="total_bill",hue="smoker", data=tips)
g = sns.catplot(x="sex", y="total_bill",
                hue="smoker", col="time",
                data=tips)
g = sns.catplot(x="sex", y="total_bill",
                hue="smoker", col="time",
                data=tips, kind="box")
g = sns.catplot(x="sex", y="total_bill",
                hue="smoker", col="time",
                data=tips, kind="box",
                height=4, aspect=.7);