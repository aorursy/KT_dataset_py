import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-training.csv")

df.head()
sns.countplot(x="SeriousDlqin2yrs", data=df)

print("Proportion of People Who Defaulted: {}".format(df["SeriousDlqin2yrs"].sum() / len(df)))
null_val_sums = df.isnull().sum()

pd.DataFrame({"Column": null_val_sums.index, "Number of Null Values": null_val_sums.values,

             "Proportion": null_val_sums.values / len(df) })
df["RevolvingUtilizationOfUnsecuredLines"].describe()
sns.distplot(df["RevolvingUtilizationOfUnsecuredLines"])
default_prop = []

for i in range(int(df["RevolvingUtilizationOfUnsecuredLines"].max())):

    temp_ = df.loc[df["RevolvingUtilizationOfUnsecuredLines"] >= i]

    default_prop.append([i, temp_["SeriousDlqin2yrs"].mean()])

default_prop
sns.lineplot(x=[i[0] for i in default_prop], y=[i[1] for i in default_prop])

plt.title("Proportion of Defaulters As Minimum RUUL Increases")
print("Proportion of Defaulters with Total Amount of Money Owed Not Exceeding Total Credit Limit: {}"\

     .format(df.loc[(df["RevolvingUtilizationOfUnsecuredLines"] >= 0) & (df["RevolvingUtilizationOfUnsecuredLines"] <= 1)]["SeriousDlqin2yrs"].mean()))
print("Proportion of Defaulters with Total Amount of Money Owed Not Exceeding or Equal to 13 times of Total Credit Limit:\n{}"\

     .format(df.loc[(df["RevolvingUtilizationOfUnsecuredLines"] >= 0) & (df["RevolvingUtilizationOfUnsecuredLines"] < 13)]["SeriousDlqin2yrs"].mean()))
df["age"].describe()
sns.distplot(df["age"])
sns.distplot(df.loc[df["SeriousDlqin2yrs"] == 0]["age"])
sns.distplot(df.loc[df["SeriousDlqin2yrs"] == 1]["age"])
late_pay_cols = ["NumberOfTimes90DaysLate", "NumberOfTime60-89DaysPastDueNotWorse",

                "NumberOfTime30-59DaysPastDueNotWorse"]

df["NumberOfTimes90DaysLate"].value_counts().sort_index()
df["NumberOfTime60-89DaysPastDueNotWorse"].value_counts().sort_index()
df["NumberOfTime30-59DaysPastDueNotWorse"].value_counts().sort_index()
df.loc[df["NumberOfTimes90DaysLate"] > 17][late_pay_cols].describe()
distinct_triples_counts = dict()

for arr in df.loc[df["NumberOfTimes90DaysLate"] > 17][late_pay_cols].values:

    triple = ",".join(list(map(str, arr)))

    if triple not in distinct_triples_counts:

        distinct_triples_counts[triple] = 0

    else:

        distinct_triples_counts[triple] += 1

distinct_triples_counts
df["DebtRatio"].describe()
df["DebtRatio"].quantile(0.95)
df.loc[df["DebtRatio"] > df["DebtRatio"].quantile(0.95)][["DebtRatio", "MonthlyIncome", "SeriousDlqin2yrs"]].describe()
len(df[(df["DebtRatio"] > df["DebtRatio"].quantile(0.95)) & (df['SeriousDlqin2yrs'] == df['MonthlyIncome'])])
df.loc[df["DebtRatio"] > df["DebtRatio"].quantile(0.95)]["MonthlyIncome"].value_counts()
print("Number of people who owe around 2449 or more times what they own and have same values for MonthlyIncome and SeriousDlqin2yrs: {}"\

     .format(len(df.loc[(df["DebtRatio"] > df["DebtRatio"].quantile(0.95)) & (df["MonthlyIncome"] == df["SeriousDlqin2yrs"])])))
df["DebtRatio"].quantile(0.975)
df.loc[df["DebtRatio"] > df["DebtRatio"].quantile(0.975)][["DebtRatio", "MonthlyIncome", "SeriousDlqin2yrs"]].describe()
len(df[(df["DebtRatio"] > df["DebtRatio"].quantile(0.975)) & (df['SeriousDlqin2yrs'] == df['MonthlyIncome'])])
df.loc[df["DebtRatio"] > df["DebtRatio"].quantile(0.975)]["MonthlyIncome"].value_counts()
print("Number of people who owe around 3490 or more times what they own and have same values for MonthlyIncome and SeriousDlqin2yrs: {}"\

     .format(len(df.loc[(df["DebtRatio"] > df["DebtRatio"].quantile(0.975)) & (df["MonthlyIncome"] == df["SeriousDlqin2yrs"])])))
sns.distplot(df["MonthlyIncome"].dropna())
df["MonthlyIncome"].describe()
sns.distplot(df.loc[df["DebtRatio"] <= df["DebtRatio"].quantile(0.975)]["MonthlyIncome"].dropna())
df["NumberOfOpenCreditLinesAndLoans"].describe()
df["NumberOfOpenCreditLinesAndLoans"].value_counts()
sns.distplot(df["NumberOfOpenCreditLinesAndLoans"])
df["NumberRealEstateLoansOrLines"].describe()
df["NumberRealEstateLoansOrLines"].value_counts()
sns.countplot(x="NumberRealEstateLoansOrLines", data=df.loc[df["NumberRealEstateLoansOrLines"] <= 10])
df.loc[df["NumberRealEstateLoansOrLines"] > 13]["SeriousDlqin2yrs"].describe()
df["NumberOfDependents"].describe()
df["NumberOfDependents"].value_counts()
df.loc[df["NumberOfDependents"] <= 10]["SeriousDlqin2yrs"].describe()
sns.countplot(x="NumberOfDependents", data=df.loc[df["NumberOfDependents"] <= 10])