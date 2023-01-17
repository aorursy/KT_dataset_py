import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statistics

from scipy import stats

from matplotlib import pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))



from scipy.stats import kde
df = pd.read_csv("../input/murder-rates-by-states/state.csv")

df.head(10)
np.mean(df["Population"])
round(stats.trim_mean(df["Population"], 0.1), 2)
np.median(df["Population"])
round(np.average(df["Murder.Rate"], weights = df["Population"]),2 )
# l = []



# for i in range(len(df.Population)):

#     l.extend([df["Murder.Rate"][i]]*df.Population[i])

    

# wm = np.median(l)

# wm

# Result: 4,

# Code execution time: 34 seconds.
loc_pop = stats.trim_mean(df["Population"], 0.1)

dev_pop = []

for i in range(len(df.Population)):

    dev_pop.append(df["Population"][i]-loc_pop)

print("Location: ", loc_pop)

print("***************** \n")

dev_pop[:10]
loc_mrd = stats.trim_mean(df["Murder.Rate"], 0.1)

dev_mrd = []

for i in range(len(df["Murder.Rate"])):

    dev_mrd.append(round(df["Murder.Rate"][i]-loc_mrd, 2))

print("Location: ", round(loc_mrd,2))

print("***************** \n")

dev_mrd[:10]
var_pop = np.var(df["Population"])

var_pop
var_mrd = np.var(df["Murder.Rate"])

round(var_mrd, 2)
std_pop = np.std(df["Population"])

print("Standard Devation of Population: ", round(std_pop, 2))
std_mrd = np.std(df["Murder.Rate"])

print("Standard Devation of Murder Rates: ", round(std_mrd, 2))
mad_pop = df["Population"].mad()

round(mad_pop, 2)
mad_mrd = df["Murder.Rate"].mad()

round(mad_mrd, 2)
def mad(l):

    ml =[]

    med = statistics.median(l)

    for i in l:

        ml.append(abs(i - med))

    return statistics.median(ml)



MAD_pop = mad(df["Population"])

MAD_pop
MAD_mrd = mad(df["Murder.Rate"])

round(MAD_mrd, 2)
rng_pop = max(df["Population"]) - min(df["Population"])

rng_pop
from statistics import median



size = len(df["Population"])

nums = sorted(df["Population"])



q2 = median(nums)

if len(nums) % 2 == 1:

    q1 = median(nums[:size//2])

    q3 = median(nums[(size//2)+1:])

else:

    q1 = median(nums[:(size//2)+1])

    q3 = median(nums[size//2:])



print("Q1: ", q1)

print("Q2: ", q2)

print("Q3: ", q3)

print("IQR: ", q3-q1)

from statistics import median



size = len(df["Murder.Rate"])

nums = sorted(df["Murder.Rate"])



q2 = median(nums)

if len(nums) % 2 == 1:

    q1 = median(nums[:size//2])

    q3 = median(nums[(size//2)+1:])

else:

    q1 = median(nums[:(size//2)+1])

    q3 = median(nums[size//2:])



print("Q1: ", q1)

print("Q2: ", q2)

print("Q3: ", q3)

print("IQR: ", round(q3-q1, 2))
plt.figure(figsize = (5,5), dpi = 150)

plt.boxplot(df["Population"])

plt.ylabel("Population (millions)")
intervals = pd.cut(df["Population"], 10)

df["Interval"] = intervals

df.head(3)
n_of_states_in_interv = df["Interval"].value_counts()



ranges = np.array(n_of_states_in_interv.index.values)

count = np.array(n_of_states_in_interv.values)



states_in_interv = {}

for i in ranges:

    states_in_interv[i] = []

for i in range(50):

    states_in_interv[df.iloc[i, 4]].append(df.iloc[i, 3])

    

states_in_interv = list(states_in_interv.values())

states_in_interv = [",".join(l) for l in states_in_interv]
pop_freq = pd.DataFrame({"Range": ranges, 

                         "Count": count, 

                         "States": states_in_interv}).sort_values("Range").set_index(np.arange(1, 11))

pop_freq
plt.figure(figsize = (5, 5), dpi = 150)

plt.hist(df["Population"], edgecolor = "k")

plt.xlabel("Population (millions)")

plt.ylabel("Frequency")
plt.figure(figsize = (5, 5), dpi = 150)

sns.distplot(df["Murder.Rate"], hist=True, kde=True, 

             color = 'darkblue', bins = 10,

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})



plt.ylabel("Density")
dfw = pd.read_csv("../input/percentage-of-delays-by-cause-at-dfw/dfw_airline.csv")

dfw
plt.figure(dpi = 150)

plt.bar(dfw.columns, dfw.iloc[0])

plt.grid()
plt.figure(dpi = 150)

plt.pie(dfw.iloc[0], labels = dfw.columns, autopct='%1.0f%%', explode=[0.02,0.03,0,0,0.04], colors = ["gold", "cyan", "magenta"])
spdata = pd.read_csv("../input/stock-price/sp500_data.csv")

spdata.head(3)
spsec = pd.read_csv("../input/stock-price/sp500_sectors.csv")

spsec.head(3)
telecom = list(spsec[spsec["sector"] == "telecommunications_services"]["symbol"].values)

telecom
latest_spdata = spdata[spdata["Unnamed: 0"]>"2012-07-01"]

telecom = latest_spdata[['T', 'CTL', 'FTR', 'VZ', 'LVLT']]

telecom.head(5)
telecom.corr()
plt.figure(dpi=150)

plt.matshow(telecom.corr(), fignum=1)

plt.colorbar()
plt.figure(dpi = 150)

plt.scatter(telecom["T"], telecom["VZ"], marker = ".")

plt.xlabel("ATT stock")

plt.ylabel("Verizon stock")
lcl = pd.read_csv("../input/letterofcredit-loans/lc_loans.csv")

lcl.head(5)
pd.crosstab(lcl["grade"], lcl["status"], margins = True)
tax = pd.read_csv("../input/houses-tax-data/kc_tax.csv")

tax.describe()
# def rmv_outliers(lst1, lst2):

#     """This Function is removing outliers and 

#     returning two lists with the same length.

#     """

#     lst1, lst2 = lst1.dropna(), lst2.dropna()

#     d = {key: value for key, value in zip(lst1, lst2)}

    

#     l1_q1, l1_q3 = np.percentile(lst1, [25, 75])

#     l2_q1, l2_q3 = np.percentile(lst2, [25, 75])

    

#     l1_iqr = l1_q3 - l1_q1

#     l2_iqr = l2_q3 - l2_q1

    

#     l1_lower_bound = l1_q1 - (1.5 * l1_iqr)

#     l1_upper_bound = l1_q3 + (1.5 * l1_iqr)

    

#     l2_lower_bound = l2_q1 - (1.5 * l2_iqr)

#     l2_upper_bound = l2_q3 + (1.5 * l2_iqr)

  

#     for key in d.copy().keys():

#         if key < l1_lower_bound or key > l1_upper_bound:

#             del d[key]

#     for key, value in d.copy().items():

#         if value < l2_lower_bound or value > l2_upper_bound:

#             del d[key] 

#     return list(d.keys()), list(d.values())

    

# SqFtTotLiving, TaxAssessedValue = rmv_outliers(tax["SqFtTotLiving"], tax["TaxAssessedValue"])

# SqFtTotLiving, TaxAssessedValue = pd.Series(SqFtTotLiving), pd.Series(TaxAssessedValue)
SqFtTotLiving = tax[(tax["TaxAssessedValue"]<=600000) & (tax["SqFtTotLiving"] <= 4000)]["SqFtTotLiving"] 

TaxAssessedValue = tax[(tax["TaxAssessedValue"]<=600000) & (tax["SqFtTotLiving"] <= 4000)]["TaxAssessedValue"]

SqFtTotLiving, TaxAssessedValue = pd.Series(SqFtTotLiving), pd.Series(TaxAssessedValue)



plt.figure(dpi = 300)

plt.hexbin(SqFtTotLiving, TaxAssessedValue, edgecolor = "k", gridsize = 40)

plt.colorbar()



plt.title("Hexagonal binning for tax-assesed value versus square feet.\n")

plt.xlabel("Square Feet")

plt.ylabel("Tax-Assessed Value")
x, y = SqFtTotLiving, TaxAssessedValue

nbins = 40



k = kde.gaussian_kde((SqFtTotLiving, TaxAssessedValue))

xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]

zi = k(np.vstack([xi.flatten(), yi.flatten()]))



plt.figure(dpi = 300)

plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap="binary")



plt.contour(xi, yi, zi.reshape(xi.shape))
airline_stats = pd.read_csv("../input/airline-delays/airline_stats.csv")

airline_stats.head()
plt.figure(dpi = 500)

plt.ylim(-2, 50)



sns.boxplot(x = "airline", y = "pct_carrier_delay", data = airline_stats)



plt.title("Boxplot of percent of airline delays by carrier")

plt.xlabel("Airline")

plt.ylabel("Daily % of Delayed Flights")
plt.figure(dpi = 500)

plt.ylim(-2, 50)



sns.violinplot(x = "airline", y = "pct_carrier_delay", data = airline_stats)



plt.title("Violin plot of percent of airline delays by carrier")

plt.xlabel("Airline")

plt.ylabel("Daily % of Delayed Flights")
tax = tax[(tax["TaxAssessedValue"]<=600000) & (tax["SqFtTotLiving"] <= 4000)]
fig, axs = plt.subplots(2,2, dpi = 150, figsize = (12,12))

axs[0][0].hexbin(tax[tax["ZipCode"] == 98105]["SqFtTotLiving"], tax[tax["ZipCode"] == 98105]["TaxAssessedValue"], label = "98105", edgecolor = "k", gridsize = 40)

axs[0][1].hexbin(tax[tax["ZipCode"] == 98108]["SqFtTotLiving"], tax[tax["ZipCode"] == 98108]["TaxAssessedValue"], edgecolor = "k", gridsize = 40)

axs[1][0].hexbin(tax[tax["ZipCode"] == 98126]["SqFtTotLiving"], tax[tax["ZipCode"] == 98126]["TaxAssessedValue"], edgecolor = "k", gridsize = 40)

axs[1][1].hexbin(tax[tax["ZipCode"] == 98188]["SqFtTotLiving"], tax[tax["ZipCode"] == 98188]["TaxAssessedValue"], edgecolor = "k", gridsize = 40)



fig.suptitle("Tax-assessed value versus finished square feet by zip code", fontsize = 20)



axs[0][0].set_title("98105")

axs[0][1].set_title("98108")

axs[1][0].set_title("98126")

axs[1][1].set_title("98188")



fig.text(0.5, 0.04, "Tax Assessed Value", ha='center', fontsize = 16)

fig.text(0.04, 0.5, "Finished Square Feet", va='center', rotation='vertical', fontsize = 16)
