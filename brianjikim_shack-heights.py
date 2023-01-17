import pandas as pd
import seaborn as sns
df = pd.read_csv("../input/Shacknews Height - Sheet1.csv")
df.head()
df.rename(columns={"Total Inches": "HEIGHT (INCH)", "Weight": "WEIGHT (LB)"}, inplace=True)
df.info()
# data exploration
sns.pairplot(df.iloc[:, 2:])
df[(df.iloc[:, 6] > 2.5) | (df["WEIGHT (LB)"] > 500) | (df["HEIGHT (M)"] < 1)]
df = df.drop([25, 66, 79, 81])
sns.pairplot(df.iloc[:, 2:])
df[(df.iloc[:, 6] > 0.0375) | (df.iloc[:, 6] < 0.015) | (df["HEIGHT (M)"] > 2) | (df["HEIGHT (M)"] < 1.5)]
df = df.drop([6, 8, 14, 19, 85])
# let's look again and hopefully data looks good
sns.pairplot(df.iloc[:, 2:])
df.iloc[:, 2:].describe()