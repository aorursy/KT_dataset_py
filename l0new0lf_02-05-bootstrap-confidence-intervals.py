import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



np.random.seed(123)
class statistic:

    mean           = "mean"

    standard_dev   = "std-dev"

    median         = "median"

    percentile_95  = "precentile_95"



    

def getCI(rvs, x, k=10000, n=50, statistic=statistic.mean):

    """returns upper and lower limits of x% C.I of rvs"""

    m = len(rvs)

    

    # checks

    if n>m: raise Exception("n must be less than or eq to m")

    if (k%10 != 0): raise Exception("k preferrably multiple of 10")

    if type(rvs)!= np.ndarray: raise Exception("rvs must be numpy array")

    

    # 1. Sample k number of n sized random observations with replacement

    k_samples_of_size_n = []

    for _ in range(0, k):

        n_random_idxs_w_replacement = np.random.choice(m, size=n) # with replacement

        sample = rvs[n_random_idxs_w_replacement]

        k_samples_of_size_n.append(sample)

        

    # 2. Find `the-statistic` for all k samples

    k_statistics = []

    if statistic == "mean":

        for sample in k_samples_of_size_n:

            statistic = np.mean(sample)

            k_statistics.append(statistic)

    elif statistic == "median":

        for sample in k_samples_of_size_n:

            statistic = np.median(sample)

            k_statistics.append(statistic)

    elif statistic == "std-dev":

        for sample in k_samples_of_size_n:

            statistic = np.std(sample)

            k_statistics.append(statistic)

    elif statistic == "precentile_95":

        for sample in k_samples_of_size_n:

            statistic = np.percentile(sample, 95)

            k_statistics.append(statistic)

            

    # 3. order k-statistics in ascending order

    asc_k_statistics = sorted(k_statistics)

    

    # 4. find lower limit and upper limit and return

    gap = (k - (x/100)*k)/2

    l_limit_idx = int(gap)

    u_limit_idx = int(k-gap)

    

    # return range of x% C.I

    return (asc_k_statistics[l_limit_idx], asc_k_statistics[u_limit_idx])
# gen rvs (sample from a population)

mu, sigma = 10, 20

global_population = np.random.normal(mu, sigma, 1000)

rvs = np.random.choice(global_population, 500)



# config params

conf_percent = 99

num_samples  = 10000

sample_size  = 100 # <=len(rvs) i.e <=1000



# predict C.I for mean

mean_ci_99_percent = getCI(rvs, conf_percent, k=num_samples, n=sample_size, statistic=statistic.mean)

print(f"Predicted mean range from bootstrap {conf_percent}% C.I: ", mean_ci_99_percent)

print("Actual population mean: ", np.mean(global_population))
# predict C.I for median

median_ci_99_percent = getCI(rvs, conf_percent, k=num_samples, n=sample_size, statistic=statistic.median)

print(f"Predicted median range from bootstrap {conf_percent}% C.I: ", median_ci_99_percent)

print("Actual population median: ", np.median(global_population))
# predict C.I for std-dev

std_ci_99_percent = getCI(rvs, conf_percent, k=num_samples, n=sample_size, statistic=statistic.standard_dev)

print(f"Predicted standard_dev range from bootstrap {conf_percent}% C.I: ", std_ci_99_percent)

print("Actual population standard_dev: ", np.std(global_population))
# predict C.I for 95th percentile

percetile95_ci_99_percent = getCI(rvs, conf_percent, k=num_samples, n=sample_size, statistic=statistic.percentile_95)

print(f"Predicted 95th percentile range from bootstrap {conf_percent}% C.I: ", percetile95_ci_99_percent)

print("Actual 95th percentile of global_population: ", np.percentile(global_population, 95))