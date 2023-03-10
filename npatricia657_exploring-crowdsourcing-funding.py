# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_loan = pd.read_csv("../input/kiva_loans.csv")
print(df_loan.describe())
print(df_loan.info())
print(df_loan.head(5))
df_mpi = pd.read_csv('../input/kiva_mpi_region_locations.csv')
print(df_mpi.describe())
print(df_mpi.info())
print(df_mpi.head(5))
df_region = pd.read_csv("../input/loan_themes_by_region.csv")
print(df_region.describe())
print(df_region.info())
print(df_region.head(5))
df_theme = pd.read_csv("../input/loan_theme_ids.csv")
print(df_theme.describe())
print(df_theme.info())
print(df_theme.head(5))
df_missing = df_loan.copy()
df_missing = df_missing.T
loan_missed = df_missing.isnull().sum(axis=1)
df_missing["valid_count"] = (len(df_missing.columns)-loan_missed) / len(df_missing.columns)
df_missing["na_count"] = loan_missed / len(df_missing.columns)

df_missing[["na_count","valid_count"]].sort_values("na_count", ascending=True).plot.barh(stacked=True,figsize=(12,10),color=["c","y"])
plt.title("Loan missing data")
fig, ax = plt.subplots(figsize=(12,8))
plt.subplot(211)
df_loan.country.value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.ylabel("Country")
plt.title("Number of loans in different countries and regions")
plt.subplot(212)
df_loan.region.value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.xlabel("Number of loans")
plt.ylabel("Region")
fig, ax = plt.subplots(figsize=(12,8))
plt.subplot(211)
df_loan.sector.value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.ylabel("Sector")
plt.title("Number of loans for different sectors and activities")
plt.subplot(212)
df_loan.activity.value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.ylabel("Activity")
plt.xlabel("Number of loans")
fig, ax = plt.subplots(figsize=(12,10))
plt.subplot(221)
term = df_loan.term_in_months.value_counts().nlargest(10)
plt.bar(np.arange(10),term[sorted(term.index)].values)
plt.xticks(np.arange(10),sorted(term.index))
plt.xlabel("Term in months")
plt.ylabel("Number of loans")
plt.subplot(222)
lender = df_loan.lender_count.value_counts().nlargest(10)
plt.bar(np.arange(10),lender[sorted(lender.index)].values)
plt.xticks(np.arange(10),sorted(lender.index))
plt.xlabel("Lender count")
df_loan["borrower_genders"] = df_loan["borrower_genders"].astype(str)
gender = []
for g in df_loan["borrower_genders"].values:
    if str(g)!="nan":
        gender.extend([ lst.strip() for lst in g.split(",")])
gd = pd.Series(gender).value_counts()
plt.bar([0,1],gd.values,color=["c","y"])
plt.xticks([0,1],["female","male"])
plt.ylabel("Number of loans")
plt.xlabel("Sex")
df_loan.repayment_interval.value_counts().plot(kind="pie", autopct="%1.1f%%")
sn.distplot(df_loan["funded_amount"],hist=False)
sn.distplot(df_loan["loan_amount"],hist=False)
df_loan.index = pd.to_datetime(df_loan.posted_time)
plt.figure(figsize=(8,6))
ax = df_loan.loan_amount.resample("w").sum().plot()
ax = df_loan.funded_amount.resample("w").sum().plot()
ax.set_ylabel("Amount in $")
ax.set_xlabel("Month-Year")
ax.set_xlim(pd.to_datetime(df_loan.posted_time.min()),pd.to_datetime(df_loan.posted_time.max()))
ax.legend(["loan","funded"])
plt.title("Trend loan vs funded amount")
df_loan.index = pd.to_datetime(df_loan.disbursed_time)
plt.figure(figsize=(8,6))
ax = df_loan.loan_amount.resample("w").sum().plot()
ax = df_loan.funded_amount.resample("w").sum().plot()
ax.set_ylabel("Amount in $")
ax.set_xlabel("Month-Year")
ax.set_xlim(pd.to_datetime(df_loan.posted_time.min()),pd.to_datetime(df_loan.posted_time.max()))
ax.legend(["loan","funded"])
plt.title("Disbursed loan vs funded amount")
country = df_loan.groupby("country")[["funded_amount","loan_amount"]].sum()
funded = country["funded_amount"].sort_values(ascending=False).head(20)
ind = np.arange(20)
width = 0.3
fig, ax = plt.subplots(figsize=(12,8))
ax.bar(ind-width/2, funded, width=width, color="SkyBlue", label="Funded amount")
ax.bar(ind+width/2, country.loc[funded.index,"loan_amount"], width=width, color="IndianRed", label="Loan amount")
ax.set_xticks(ind)
ax.set_xticklabels(funded.index,rotation=90)
ax.set_xlabel("Country")
ax.set_ylabel("Total amount")
ax.set_title("Countries with the highest funded amount vs loan amount")
sector = df_loan.groupby("sector")[["funded_amount","loan_amount"]].sum()
funded = sector["funded_amount"].sort_values(ascending=False).head(10)
ind = np.arange(10)
width = 0.3
fig, ax = plt.subplots(figsize=(12,8))
ax.bar(ind-width/2, funded, width=width, color="SkyBlue", label="Funded amount")
ax.bar(ind+width/2, sector.loc[funded.index,"loan_amount"], width=width, color="IndianRed", label="Loan amount")
ax.set_xticks(ind)
ax.set_xticklabels(funded.index,rotation=90)
ax.set_xlabel("Sector")
ax.set_ylabel("Total amount")
ax.set_title("Sectors with the highest funded amount vs loan amount")
activity = df_loan.groupby("activity")[["funded_amount","loan_amount"]].sum()
funded = activity["funded_amount"].sort_values(ascending=False).head(10)
ind = np.arange(10)
width = 0.3
fig, ax = plt.subplots(figsize=(12,8))
ax.bar(ind-width/2, funded, width=width, color="SkyBlue", label="Funded amount")
ax.bar(ind+width/2, activity.loc[funded.index,"loan_amount"], width=width, color="IndianRed", label="Loan amount")
ax.set_xticks(ind)
ax.set_xticklabels(funded.index,rotation=90)
ax.set_xlabel("Activity")
ax.set_ylabel("Total amount")
ax.set_title("Activities with the highest funded amount vs loan amount")
df_loan["date"] = pd.to_datetime(df_loan["date"])
df_loan["year"] = df_loan.date.dt.year
cc = df_loan.groupby(["country","year"])["funded_amount"].mean().unstack()
cc = cc.sort_values([2017],ascending=False)
cc = cc.fillna(0)
sn.heatmap(cc,cmap="Blues")
ss = df_loan.groupby(["sector","year"])["funded_amount"].mean().unstack()
ss = ss.sort_values([2017],ascending=False)
ss = ss.fillna(0)
sn.heatmap(ss,cmap="Reds")
aa = df_loan.groupby(["activity","year"])["funded_amount"].mean().unstack()
aa = aa.sort_values([2017],ascending=False)
aa = aa.fillna(0)
sn.heatmap(aa,cmap="Greens")
sn.boxplot(x="year",y="funded_amount",data=df_loan)
plt.figure(figsize=(12,8))
df_loan[df_loan["country"]=="Philippines"]["sector"].value_counts().sort_values(ascending=True).head(20).plot.barh(stacked=True)
plt.ylabel("Sector")
plt.xlabel("Number of loans")
plt.title("Sectors in Philippines with highest loan")
fig, ax = plt.subplots(figsize=(12,8))
plt.subplot(221)
pt = df_loan[df_loan["country"]=="Philippines"]["term_in_months"].value_counts().nlargest(10)
plt.bar(np.arange(10),pt[sorted(pt.index)].values)
plt.xticks(np.arange(10),sorted(pt.index))
plt.xlabel("Term in months")
plt.ylabel("Number of loans")
plt.subplot(222)
lt = df_loan[df_loan["country"]=="Philippines"]["lender_count"].value_counts().nlargest(10)
plt.bar(np.arange(10),lt[sorted(lt.index)].values)
plt.xticks(np.arange(10),sorted(lt.index))
plt.ylabel("Number of loans")
plt.xlabel("Activity")
gender = []
for g in df_loan[df_loan["country"]=="Philippines"]["borrower_genders"].values:
    if str(g)!="nan":
        gender.extend([ lst.strip() for lst in g.split(",")])
gd = pd.Series(gender).value_counts()
plt.bar([0,1],gd.values,color=["c","y"])
plt.xticks([0,1],["female","male"])
plt.ylabel("Number of loans")
plt.xlabel("Sex")
plt.title("Genders in Philippines")
df_loan[df_loan["country"]=="Philippines"]["repayment_interval"].value_counts().plot(kind="pie", autopct="%1.1f%%")
df_loan.index = pd.to_datetime(df_loan.disbursed_time)
plt.figure(figsize=(8,6))
ax = df_loan[df_loan["country"]=="Philippines"]["loan_amount"].resample("w").sum().plot()
ax = df_loan[df_loan["country"]=="Philippines"]["funded_amount"].resample("w").sum().plot()
ax.set_ylabel("Amount in $")
ax.set_xlabel("Month-Year")
ax.set_xlim(pd.to_datetime(df_loan.posted_time.min()),pd.to_datetime(df_loan.posted_time.max()))
ax.legend(["loan","funded"])
plt.title("Disbursed loan vs funded amount in Philippines")
plt.figure(figsize=(12,8))
df_loan[df_loan["country"]=="Kenya"]["sector"].value_counts().sort_values(ascending=True).head(20).plot.barh(stacked=True)
plt.ylabel("Sector")
plt.xlabel("Number of loans")
plt.title("Sectors in Kenya with highest loan")
fig, ax = plt.subplots(figsize=(12,8))
plt.subplot(221)
pt = df_loan[df_loan["country"]=="Kenya"]["term_in_months"].value_counts().nlargest(10)
plt.bar(np.arange(10),pt[sorted(pt.index)].values)
plt.xticks(np.arange(10),sorted(pt.index))
plt.xlabel("Term in months")
plt.ylabel("Number of loans")
plt.subplot(222)
lt = df_loan[df_loan["country"]=="Kenya"]["lender_count"].value_counts().nlargest(10)
plt.bar(np.arange(10),lt[sorted(lt.index)].values)
plt.xticks(np.arange(10),sorted(lt.index))
plt.ylabel("Number of loans")
plt.xlabel("Activity")
gender = []
for g in df_loan[df_loan["country"]=="Kenya"]["borrower_genders"].values:
    if str(g)!="nan":
        gender.extend([ lst.strip() for lst in g.split(",")])
gd = pd.Series(gender).value_counts()
plt.bar([0,1],gd.values,color=["c","y"])
plt.xticks([0,1],["female","male"])
plt.ylabel("Number of loans")
plt.xlabel("Sex")
plt.title("Genders in Kenya")
df_loan[df_loan["country"]=="Kenya"]["repayment_interval"].value_counts().plot(kind="pie", autopct="%1.1f%%")
df_loan.index = pd.to_datetime(df_loan.disbursed_time)
plt.figure(figsize=(8,6))
ax = df_loan[df_loan["country"]=="Kenya"]["loan_amount"].resample("w").sum().plot()
ax = df_loan[df_loan["country"]=="Kenya"]["funded_amount"].resample("w").sum().plot()
ax.set_ylabel("Amount in $")
ax.set_xlabel("Month-Year")
ax.set_xlim(pd.to_datetime(df_loan.posted_time.min()),pd.to_datetime(df_loan.posted_time.max()))
ax.legend(["loan","funded"])
plt.title("Disbursed loan vs funded amount in Kenya")
plt.figure(figsize=(12,8))
df_mpi.groupby("country")["MPI"].mean().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.xlabel("MPI")
plt.title("Average MPI")
plt.figure(figsize=(12,8))
df_mpi.groupby("world_region")["MPI"].mean().sort_values(ascending=True).plot.barh(stacked=True)
plt.xlabel("MPI")
plt.title("Average MPI")
df_miss_reg = df_region.copy()
df_miss_reg = df_miss_reg.T
reg_missed = df_miss_reg.isnull().sum(axis=1)
df_miss_reg["valid_count"] = (len(df_miss_reg.columns)-reg_missed) / len(df_miss_reg.columns)
df_miss_reg["na_count"] = reg_missed / len(df_miss_reg.columns)

df_miss_reg[["na_count","valid_count"]].sort_values("na_count", ascending=True).plot.barh(stacked=True,figsize=(12,10),color=["c","y"])
plt.title("Loan theme missing data")
plt.figure(figsize=(12,8))
df_region["Field Partner Name"].value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.xlabel("Number of themes")
plt.ylabel("Kiva's Partner")
plt.title("Top 20 Kiva's partner")
plt.figure(figsize=(12,8))
df_region["Loan Theme Type"].value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.xlabel("Number of loans")
plt.ylabel("Loan Theme")
plt.title("Top 20 loan types")
plt.figure(figsize=(12,8))
df_region["sector"].value_counts().sort_values(ascending=True).plot.barh(stacked=True)
plt.xlabel("Number of loans")
plt.ylabel("Sector")
plt.title("Top sectors")
plt.figure(figsize=(12,8))
df_region["country"].value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.xlabel("Number of loans")
plt.ylabel("Country")
plt.title("Top 20 Countries")
plt.figure(figsize=(12,8))
df_region["region"].value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.xlabel("Number of loans")
plt.ylabel("Region")
plt.title("Top 20 Regions")
df_region["forkiva"].value_counts().plot(kind="pie", autopct="%1.1f%%")
df_region.groupby("forkiva")["amount"].sum().plot.bar()
plt.xticks(rotation=360)
plt.ylabel("Total Amount in billion")
plt.title("Total loan amount given by kiva's non-partner vs partner")
fig, ax = plt.subplots(figsize=(12,10))
plt.subplot(211)
df_region[df_region["forkiva"]=="No"]["Field Partner Name"].value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.ylabel("Field Partner")
plt.subplot(212)
df_region[df_region["forkiva"]=="Yes"]["Field Partner Name"].value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.ylabel("Field Partner")
plt.xlabel("Number of loans")
fig, ax = plt.subplots(figsize=(12,10))
plt.subplot(211)
df_region[df_region["forkiva"]=="No"]["country"].value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.ylabel("Country")
plt.title("Country for non partner")
plt.subplot(212)
df_region[df_region["forkiva"]=="Yes"]["country"].value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.ylabel("Country")
plt.xlabel("Number of loans")
plt.title("Country for Kiva's partner")
fig, ax = plt.subplots(figsize=(12,10))
plt.subplot(211)
df_region[df_region["forkiva"]=="No"]["sector"].value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.ylabel("Sector")
plt.title("Sector for non partner")
plt.subplot(212)
df_region[df_region["forkiva"]=="Yes"]["sector"].value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.ylabel("Sector")
plt.xlabel("Number of loans")
plt.title("Sector for Kiva's partner")
fig, ax = plt.subplots(figsize=(12,10))
plt.subplot(211)
df_region[df_region["forkiva"]=="No"]["Loan Theme Type"].value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.ylabel("Loan Type")
plt.title("Loan theme for non partner")
plt.subplot(212)
df_region[df_region["forkiva"]=="Yes"]["Loan Theme Type"].value_counts().nlargest(20).sort_values(ascending=True).plot.barh(stacked=True)
plt.ylabel("Loan Type")
plt.xlabel("Number of loans")
plt.title("Loan theme for Kiva's partner")