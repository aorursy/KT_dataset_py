

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 



import statsmodels.api as sm

import scipy.stats.distributions as dist



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

da = pd.read_csv("/kaggle/input/nhanes/NHANES.csv")

da.head()

da["SMQ020x"] = da.SMQ020.replace({1: "Yes", 2: "No", 7: np.nan, 9: np.nan})  # np.nan represents a missing value

da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})

da["DMDCITZNx"] = da.DMDCITZN.replace({1: "Yes", 2: "No", 7: np.nan, 9: np.nan})
x = da.SMQ020x.dropna() == "Yes"

p = x.mean()

se = np.sqrt(0.4 * 0.6 / len(x))

test_stat = (p - 0.4) / se

pvalue = 2*dist.norm.cdf(-np.abs(test_stat))

print(test_stat, pvalue)
# Prints test statistic, p-value

print(sm.stats.proportions_ztest(x.sum(), len(x), 0.4)) # Normal approximation with estimated proportion in SE

print(sm.stats.proportions_ztest(x.sum(), len(x), 0.4, prop_var=0.4)) # Normal approximation with null proportion in SE



# Prints the p-value

print(sm.stats.binom_test(x.sum(), len(x), 0.4)) # Exact binomial p-value
dx = da[["SMQ020x", "RIDAGEYR", "RIAGENDRx"]].dropna()  # Drop missing values

dx = dx.loc[(dx.RIDAGEYR >= 20) & (dx.RIDAGEYR <= 25), :] # Restrict to people between 20 and 25 years old



# Summarize the data by caclculating the proportion of yes responses and the sample size

p = dx.groupby("RIAGENDRx")["SMQ020x"].agg([lambda z: np.mean(z=="Yes"), "size"])

p.columns = ["Smoke", "N"]

print(p)



# The pooled rate of yes responses, and the standard error of the estimated difference of proportions

p_comb = (dx.SMQ020x == "Yes").mean()

va = p_comb * (1 - p_comb)

se = np.sqrt(va * (1 / p.N.Female + 1 / p.N.Male))



# Calculate the test statistic and its p-value

test_stat = (p.Smoke.Female - p.Smoke.Male) / se

pvalue = 2*dist.norm.cdf(-np.abs(test_stat))

print(test_stat, pvalue)
dx_females = dx.loc[dx.RIAGENDRx=="Female", "SMQ020x"].replace({"Yes": 1, "No": 0})

dx_males = dx.loc[dx.RIAGENDRx=="Male", "SMQ020x"].replace({"Yes": 1, "No": 0})

sm.stats.ttest_ind(dx_females, dx_males) # prints test statistic, p-value, degrees of freedom
dx = da[["BPXSY1", "RIDAGEYR", "RIAGENDRx"]].dropna()

dx = dx.loc[(dx.RIDAGEYR >= 40) & (dx.RIDAGEYR <= 50) & (dx.RIAGENDRx == "Male"), :]

print(dx.BPXSY1.mean()) # prints mean blood pressure

sm.stats.ztest(dx.BPXSY1, value=120)  # prints test statistic, p-value
dx = da[["BPXSY1", "RIDAGEYR", "RIAGENDRx"]].dropna()

dx = dx.loc[(dx.RIDAGEYR >= 50) & (dx.RIDAGEYR <= 60), :]

bpx_female = dx.loc[dx.RIAGENDRx=="Female", "BPXSY1"]

bpx_male = dx.loc[dx.RIAGENDRx=="Male", "BPXSY1"]

print(bpx_female.mean(), bpx_male.mean()) # prints female mean, male mean

print(sm.stats.ztest(bpx_female, bpx_male)) # prints test statistic, p-value

print(sm.stats.ttest_ind(bpx_female, bpx_male)) # prints test statistic, p-value, degrees of freedom
dx = da[["BMXBMI", "RIDAGEYR", "RIAGENDRx"]].dropna()

da["agegrp"] = pd.cut(da.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])

da.groupby(["agegrp", "RIAGENDRx"])["BMXBMI"].agg(np.std).unstack()
for k, v in da.groupby("agegrp"):

    bmi_female = v.loc[v.RIAGENDRx=="Female", "BMXBMI"].dropna()

    bmi_female = sm.stats.DescrStatsW(bmi_female)

    bmi_male = v.loc[v.RIAGENDRx=="Male", "BMXBMI"].dropna()

    bmi_male = sm.stats.DescrStatsW(bmi_male)

    print(k)

    print("pooled: ", sm.stats.CompareMeans(bmi_female, bmi_male).ztest_ind(usevar='pooled'))

    print("unequal:", sm.stats.CompareMeans(bmi_female, bmi_male).ztest_ind(usevar='unequal'))

    print()
dx = da[["BPXSY1", "BPXSY2"]].dropna()

db = dx.BPXSY1 - dx.BPXSY2

print(db.mean())

sm.stats.ztest(db)
dx = da[["RIAGENDRx", "BPXSY1", "BPXSY2", "RIDAGEYR"]].dropna()

dx["agegrp"] = pd.cut(dx.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])

for k, g in dx.groupby(["RIAGENDRx", "agegrp"]):

    db = g.BPXSY1 - g.BPXSY2

    # print stratum definition, mean difference, sample size, test statistic, p-value

    print(k, db.mean(), db.size, sm.stats.ztest(db.values, value=0))
all_p = []

dy = dx.loc[(dx.RIDAGEYR >= 50) & (dx.RIDAGEYR <= 60), :]

for n in 100, 200, 400, 800:

    pv = []

    for i in range(500):

        dz = dy.sample(n)

        db = dz.BPXSY1 - dz.BPXSY2

        _, p = sm.stats.ztest(db.values, value=0)

        pv.append(p)

    pv = np.asarray(pv)

    all_p.append(pv)

    print((pv <= 0.05).mean())
sns.distplot(all_p[0])
sns.distplot(all_p[2])