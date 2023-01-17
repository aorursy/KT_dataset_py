#Like always we'll import libraries:



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline     



import statsmodels.api as sm



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#we'll import data now   

da = pd.read_csv('/kaggle/input/nhanes/NHANES.csv') #Data is available in my dataset

da.head()
da["SMQ020x"] = da.SMQ020.replace({1: "Yes", 2: "No", 7: np.nan, 9: np.nan})  # np.nan represents a missing value

da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})
dx = da[["SMQ020x", "RIAGENDRx"]].dropna()  # dropna drops cases where either variable is missing

pd.crosstab(dx.SMQ020x, dx.RIAGENDRx)
dz = dx.groupby(dx.RIAGENDRx).agg({"SMQ020x": [lambda x: np.mean(x=="Yes"), np.size]})     #agg is used to aggregate 

dz.columns = ["Proportion", "Total_n"] # The default column names are unclear, so we replace them here

dz
p = dz.Proportion.Female # Female proportion

n = dz.Total_n.Female # Total number of females

se_female = np.sqrt(p * (1 - p) / n)

print(se_female)



p = dz.Proportion.Male # Male proportion

n = dz["Total_n"].Male # Total number of males

se_male = np.sqrt(p * (1 - p) / n)

print(se_male)
p = dz.Proportion.Female # Female proportion

n = dz.Total_n.Female # Total number of females

lcb = p - 1.96 * np.sqrt(p * (1 - p) / n)  

ucb = p + 1.96 * np.sqrt(p * (1 - p) / n)  

print(lcb, ucb)
p = dz.Proportion.Male # Male proportion

n = dz.Total_n.Male # Total number of males

lcb = p - 1.96 * np.sqrt(p * (1 - p) / n)  

ucb = p + 1.96 * np.sqrt(p * (1 - p) / n)  

print(lcb, ucb)
# 95% CI for the proportion of females who smoke (compare to value above)

sm.stats.proportion_confint(906, 906+2066)
# 95% CI for the proportion of males who smoke (compare to value above)

sm.stats.proportion_confint(1413, 1413+1340)  
se_diff = np.sqrt(se_female**2 + se_male**2)

se_diff
d = dz.Proportion.Female - dz.Proportion.Male

lcb = d - 2*se_diff

ucb = d + 2*se_diff

print(lcb, ucb)

print(d, d)
# Calculate the smoking rates within age/gender groups

da["agegrp"] = pd.cut(da.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])

pr = da.groupby(["agegrp", "RIAGENDRx"]).agg({"SMQ020x": lambda x: np.mean(x=="Yes")}).unstack()

pr.columns = ["Female", "Male"]



# The number of people for each calculated proportion

dn = da.groupby(["agegrp", "RIAGENDRx"]).agg({"SMQ020x": np.size}).unstack()

dn.columns = ["Female", "Male"]



# Standard errors for each proportion

se = np.sqrt(pr * (1 - pr) / dn)



# Standard error for the difference in female/male smoking rates in every age band

se_diff = np.sqrt(se.Female**2 + se.Male**2)



# Standard errors for the difference in smoking rates between genders, within age bands



# The difference in smoking rates between genders

pq = pr.Female - pr.Male



x = np.arange(pq.size)

pp = sns.pointplot(x, pq.values, color='black')

sns.pointplot(x, pq - 2*se_diff)

sns.pointplot(x, pq + 2*se_diff)

pp.set_xticklabels(pq.index)

pp.set_xlabel("Age group")

pp.set_ylabel("Female - male smoking proportion")
da.groupby("RIAGENDRx").agg({"BMXBMI": np.mean})
sem_female = 7.753 / np.sqrt(2976)

sem_male = 6.253 / np.sqrt(2759)

print(sem_female, sem_male)
lcb_female = 29.94 - 1.96 * 7.753 / np.sqrt(2976)

ucb_female = 29.94 + 1.96 * 7.753 / np.sqrt(2976)

print(lcb_female, ucb_female)
female_bmi = da.loc[da.RIAGENDRx=="Female", "BMXBMI"].dropna()

sm.stats.DescrStatsW(female_bmi).zconfint_mean()
sem_diff = np.sqrt(sem_female**2 + sem_male**2)

sem_diff
bmi_diff = 29.94 - 28.78

lcb = bmi_diff - 2*sem_diff

ucb = bmi_diff + 2*sem_diff

(lcb, ucb)
# Calculate the mean, SD, and sample size for BMI within age/gender groups

da["agegrp"] = pd.cut(da.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])

pr = da.groupby(["agegrp", "RIAGENDRx"]).agg({"BMXBMI": [np.mean, np.std, np.size]}).unstack()



# Calculate the SEM for females and for males within each age band

pr["BMXBMI", "sem", "Female"] = pr["BMXBMI", "std", "Female"] / np.sqrt(pr["BMXBMI", "size", "Female"]) 

pr["BMXBMI", "sem", "Male"] = pr["BMXBMI", "std", "Male"] / np.sqrt(pr["BMXBMI", "size", "Male"]) 



# Calculate the mean difference of BMI between females and males within each age band, also  calculate

# its SE and the lower and upper limits of its 95% CI.

pr["BMXBMI", "mean_diff", ""] = pr["BMXBMI", "mean", "Female"] - pr["BMXBMI", "mean", "Male"]

pr["BMXBMI", "sem_diff", ""] = np.sqrt(pr["BMXBMI", "sem", "Female"]**2 + pr["BMXBMI", "sem", "Male"]**2) 

pr["BMXBMI", "lcb_diff", ""] = pr["BMXBMI", "mean_diff", ""] - 1.96 * pr["BMXBMI", "sem_diff", ""] 

pr["BMXBMI", "ucb_diff", ""] = pr["BMXBMI", "mean_diff", ""] + 1.96 * pr["BMXBMI", "sem_diff", ""] 



# Plot the mean difference in black and the confidence limits in blue

x = np.arange(pr.shape[0])

pp = sns.pointplot(x, pr["BMXBMI", "mean_diff", ""], color='black')

sns.pointplot(x, pr["BMXBMI", "lcb_diff", ""], color='blue')

sns.pointplot(x, pr["BMXBMI", "ucb_diff", ""], color='blue')

pp.set_xticklabels(pr.index)

pp.set_xlabel("Age group")

pp.set_ylabel("Female - male BMI difference")
print(pr)
dx = da.loc[da.RIAGENDRx=="Female", ["RIAGENDRx", "BMXBMI"]].dropna()



all_cis = []

for n in 100, 200, 400, 800:

    cis = []

    for i in range(500):

        dz = dx.sample(n)

        ci = sm.stats.DescrStatsW(dz.BMXBMI).zconfint_mean()

        cis.append(ci)

    cis = np.asarray(cis)

    mean_width = cis[:, 1].mean() - cis[:, 0].mean()

    print(n, mean_width)

    all_cis.append(cis)
ci = all_cis[0]

for j, x in enumerate(ci):

    plt.plot([j, j], x, color='grey')

    plt.gca().set_ylabel("BMI")

mn = dx.BMXBMI.mean()

plt.plot([0, 500], [mn, mn], color='red')
print(np.mean(ci[:, 1] < mn)) # Upper limit falls below the target

print(np.mean(ci[:, 0] > mn)) # Lower limit falls above the target