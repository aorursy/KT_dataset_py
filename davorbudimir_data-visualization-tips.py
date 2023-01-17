import pandas as pd
import seaborn as sns

file_path = '../input/tipsdataset/tips.csv'
tips = pd.read_csv(file_path)

print(tips.head())
print(tips.describe())
sns.distplot(tips['tip'], kde=False, bins=30)
sns.distplot(tips['tip'],hist=False, bins=10)
sns.relplot(x="total_bill", y="tip", data=tips)
sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips)
sns.relplot(x="total_bill", y="tip", kind="line", data=tips)
sns.catplot(x="sex", y="tip", data=tips)
sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips)
sns.catplot(x="smoker", y="tip", order=["No", "Yes"], data=tips)
sns.catplot(x="day", y="total_bill", kind="box", data=tips)
sns.catplot(x="day", y="total_bill", hue="sex", kind="box", data=tips)
sns.catplot(x="day", y="total_bill", hue="sex", kind="violin", data=tips)
sns.catplot(x="day", y="total_bill", hue="sex", kind="point", data=tips)
