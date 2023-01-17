%matplotlib inline
import numpy as np
import pandas as pd
import os
import seaborn as sns

from matplotlib import pyplot as plt

pd.set_option('max_columns', 150)
DATA_DIR = "../input/"
df = pd.read_csv(DATA_DIR + "application_train.csv")
df_cc = pd.read_csv(DATA_DIR + "credit_card_balance.csv")
df.head()
sns.countplot(x="NAME_CONTRACT_TYPE", hue="TARGET", data=df);
sns.countplot(hue="TARGET", x="CODE_GENDER", data=df);
sns.factorplot(x="CODE_GENDER", hue="TARGET", col="OCCUPATION_TYPE", data=df, kind="count", col_wrap=4, sharey=False);
# Very few applicants with Academic degree
df["NAME_EDUCATION_TYPE"].value_counts()
sns.factorplot(x="CODE_GENDER", hue="TARGET", col="NAME_EDUCATION_TYPE", data=df, kind="count", col_wrap=4, sharey=False)
df["OWN_CAR_OR_REALITY"] = (df["FLAG_OWN_REALTY"] == "Y") | (df["FLAG_OWN_CAR"] == "Y")
sns.factorplot(x="OWN_CAR_OR_REALITY", hue="TARGET", col="CODE_GENDER",
    data=df, kind="count", col_wrap=4, sharey=False)
df_cc.head()
df_with_cc_info = pd.merge(df, df_cc, on="SK_ID_CURR", how="right")
df_mean = df_with_cc_info.groupby(["TARGET"]).mean()
df_with_cc_info = df_with_cc_info.groupby(["SK_ID_CURR", "TARGET"]).mean().reset_index()
df_mean[["AMT_DRAWINGS_CURRENT", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_TOTAL_RECEIVABLE"]]
plt.figure(figsize=(20, 20))
for i in range(25):
    plt.subplot(5,5,i+1)
    df_cc_sample = df_cc[df_cc["SK_ID_CURR"] == df_with_cc_info[df_with_cc_info["TARGET"] == 1].sample(1)["SK_ID_CURR"].values[0]] \
        .sort_values("MONTHS_BALANCE")
    plt.plot(df_cc_sample["MONTHS_BALANCE"], df_cc_sample["AMT_DRAWINGS_CURRENT"])
plt.figure(figsize=(20, 20))
for i in range(25):
    plt.subplot(5,5,i+1)
    df_cc_sample = df_cc[df_cc["SK_ID_CURR"] == df_with_cc_info[df_with_cc_info["TARGET"] == 0].sample(1)["SK_ID_CURR"].values[0]] \
        .sort_values("MONTHS_BALANCE")
    plt.plot(df_cc_sample["MONTHS_BALANCE"], df_cc_sample["AMT_DRAWINGS_CURRENT"])
df_with_cc_info[df_with_cc_info["AMT_DRAWINGS_CURRENT"] == 0]["TARGET"].value_counts()
print("Total applications:", df["SK_ID_CURR"].nunique())
print("Total applications records in CC file(current):", df_cc["SK_ID_CURR"].nunique())
print("Matched records :", df_with_cc_info["SK_ID_CURR"].nunique())