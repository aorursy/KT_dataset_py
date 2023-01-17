import pandas as pd
from subprocess import check_output

print(check_output(["ls", "/kaggle/input/"]).decode("utf8"))
print(check_output(["ls", "/kaggle/input/loandata"]).decode("utf8"))
df = pd.read_csv("/kaggle/input/loandata/loanData.csv")
df.describe()
df["annual_inc"].describe()
df[["annual_inc"]].info()
df.columns
df.iloc[:, 3]
df["final_d"]
df.iloc[:, 3:6]
df.iloc[4]
df.iloc[2:5]
df.iloc[2:5, 2:5]
df.iloc[:, 2].name
df.columns.get_loc("issue_d")
df.count()
df.drop_duplicates()
df.drop_duplicates().count()
df.drop_duplicates(subset ="interest_rate").count()
df.count()
df.drop_duplicates(subset ="interest_rate", inplace = True)
df.count()
df_copy = pd.read_csv("/kaggle/input/loandata/loanData.csv")
df_copy.head()
df_copy["id"].count()
df_copy.drop_duplicates(subset="interest_rate", keep=False).count()
df.sort_values("annual_inc").head()
df.sort_values("annual_inc", ascending = False).head()
df.sort_values(["home_ownership", "annual_inc"])
df_int = df[df["interest_rate"] > 15.2]
df_conditioned = df[(df["interest_rate"] > 15.2) & (df["total_pymnt"] > 20000)]
df_conditioned.count()
df_conditioned.head()
df_copy.groupby("home_ownership").size()
type(df_copy.groupby("home_ownership"))
df_copy.groupby("home_ownership").groups["MORTGAGE"]
df_copy.groupby("home_ownership")["interest_rate"].max().head()
df_copy.groupby("home_ownership")["interest_rate"].mean().head()
df.groupby("home_ownership").agg({"interest_rate": ["min", "max", "mean", "median"],

                                 "annual_inc":["min", "max", "mean", "median"]})
df_copy.groupby("home_ownership").size()
df_copy.groupby(["home_ownership", "income_category"]).size()
df.isnull().sum()
df.dropna()

df.fillna(0)
import matplotlib.pyplot as plt

%matplotlib inline
ax = df['interest_rate'].plot.hist(bins=20)

ax.set_xlabel('Interest Rate Histogram')
## 2 columns

df.plot(y='recoveries', x='total_pymnt')