# Run this cell to obtain the file path.

# Use the second file path with the csv extension.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd
df=pd.read_csv("/kaggle/input/sf-salariescleaned/Salaries.csv", index_col=0)
df.head()
df.info()
print(df.isnull().sum())

df.dropna(axis=1, how="all", inplace=True)

print(df.info())
# df.index = df["Id"]

# df.drop("Id", axis=1, inplace=True)
df["BasePay"].mean()
df["OvertimePay"].max()
df[df["EmployeeName"] == "JOSEPH DRISCOLL"]["JobTitle"]
df[df["EmployeeName"] == "JOSEPH DRISCOLL"]["TotalPayBenefits"]
df[(df["TotalPayBenefits"]) == (df["TotalPayBenefits"].max())][["EmployeeName", "TotalPayBenefits"]]
df[(df["TotalPayBenefits"]) == (df["TotalPayBenefits"].min())][["EmployeeName", "TotalPayBenefits"]]
df.groupby(["Year"]).mean()["BasePay"]
df["JobTitle"].nunique()
df["JobTitle"].value_counts().head()
sum(df[df["Year"] == 2013]["JobTitle"].value_counts()==1)
def chief_finder(title):

    title = title.lower()

    if "chief" in title:

        return True

    else:

        return False

    

sum(df["JobTitle"].apply(lambda x:chief_finder(x)))
df["title_len"] = df["JobTitle"].apply(len)

df[["title_len", "TotalPayBenefits"]].corr()