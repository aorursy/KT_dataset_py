import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import random as r

from scipy import stats
titanic = pd.read_csv("../input/titanic/train.csv")
titanic.head()
titanic.info()
plt.hist(titanic["Age"])

plt.text(0.7, 0.9, f"Mean: {round(np.mean(titanic.Age), 3)}", transform = plt.gca().transAxes)

plt.title("Distribution of Age on the Titanic")

plt.show()
def central_limit_theorem(titanic, column, N, n): #N is number of samples, n is size of each sample

    

    titanic = titanic[np.isfinite(titanic["Age"])] #Removing missing data, as we have 714 known ages

    

    sample_means = []

    for i in range(N):

        sample_means.append(np.mean(r.sample(list(titanic[column]), n))) #For N times, Getting a list of sample means of size n 

    return sample_means



def clt_graph(titanic, column, N, n):

    sample_means = central_limit_theorem(titanic, column, N, n)

    true_mean_diff = round(np.mean(titanic.Age) - np.mean(sample_means), 2)

    plt.hist(sample_means)

    plt.text(0.7, 0.9, f"Mean: {round(np.mean(sample_means), 2)}", transform = plt.gca().transAxes)

    plt.text(0.7, 0.8, f"Difference: {true_mean_diff}", transform = plt.gca().transAxes)

    plt.title(f"Distribution of Sample Age Means (n = {n}; N = {N})")

    plt.show()
[clt_graph(titanic, "Age", 10000, i) for i in [2, 30, 60, 100]]
[clt_graph(titanic, "Age", i, 30) for i in [10, 50, 100, 1000]]
#The sample_mean2 paramter allows us to enter another sample mean if we want the probability of a range of two values

#The greater_than parameter allows us to either get the probability to the left or right of our z-score



def z_probability_sample_means(titanic, column, n, sample_mean, sample_mean2 = 0, greater_than = False):

    

    sample_mean_list = central_limit_theorem(titanic, column, 10000, n) #getting sample means of age, following a normal distribution by CLT.

    

    pop_mean = np.mean(sample_mean_list) #Again, this approximates the true population mean if N and n are large enough

    

    standard_error = np.std(titanic[column]) / np.sqrt(n) #Getting the standard error

    

    z_score = (sample_mean - pop_mean) / standard_error #Calculating the z-score (std's from the mean)

    

    if sample_mean2 != 0: #These statements will activate if we want a probability within a range of two sample means

        z_score2 = (sample_mean2 - pop_mean) / standard_error

        return f"Interpretation: The probability that we acquire a sample mean between {round(min(sample_mean, sample_mean2), 2)} and {round(max(sample_mean, sample_mean2), 2)} is {round((stats.norm.cdf(max(z_score, z_score2)) - stats.norm.cdf(min(z_score, z_score2))) * 100, 2)}%"

    

    if greater_than == False:

        return f"Interpretation: The probability that we acquire a sample mean less than {sample_mean} is {round(stats.norm.cdf(z_score) * 100, 2)}%" #Finally, we get the corresponding value from the z-score table

    else:

        return f"Interpretation: The probability that we acquire a sample mean greater than {sample_mean} is {round((1 - stats.norm.cdf(z_score)) * 100, 2)}%" #Since the z_score area gives us the area to the left, we must subract this area from 1 if we want a probability of a sample mean being greater than some value.
z_probability_sample_means(titanic, "Age", sample_mean = 25, n = 50, greater_than = False)
z_probability_sample_means(titanic, "Age", sample_mean = 30, n = 50, greater_than = True)
z_probability_sample_means(titanic, "Age", sample_mean = 28, sample_mean2 = 30, n = 50)
def z_confidence_interval_sample_mean(titanic, column, n, conf):

    

    titanic = titanic[np.isfinite(titanic["Age"])] #Remove missing values

    

    sample_mean = np.mean(r.sample(list(titanic[column]), n)) #getting a single sample mean

    

    standard_error = np.std(titanic[column]) / np.sqrt(n) #same formula we used previously for SE

    

    z_star = stats.norm.ppf(((1 - conf) / 2) + conf) #For any confidence level, the probability we want from our z table is halfway between that value and 1 because we just take into account both tails of the distribution

    

    CI =  sample_mean - (z_star * standard_error), sample_mean + (z_star * standard_error)

    

    return f"Interpretation: With a sample mean of {round(sample_mean, 2)}, we can be {round(conf * 100, 2)}% confident that the true population mean lies between {round(CI[0], 2)} and {round(CI[1], 2)}."

    
z_confidence_interval_sample_mean(titanic, "Age", n = 50, conf = 0.95)
z_confidence_interval_sample_mean(titanic, "Age", n = 50, conf = 0.50)
z_confidence_interval_sample_mean(titanic, "Fare", n = 50, conf = 0.80)
def t_confidence_interval_sample_mean(titanic, column, n, conf):

        

    titanic = titanic[np.isfinite(titanic["Age"])] #Remove missing values

    

    sample = r.sample(list(titanic[column]), n)

    sample_mean = np.mean(sample) #Getting sample mean and standard deviation from our sample of n <= 30

    sample_std = np.std(sample)

    

    standard_error = sample_std / np.sqrt(n) #same formula we used previously for SE

    

    t_star = stats.t.ppf(((1 - conf) / 2) + conf, n - 1) #Same process as Z table, but df = n - 1 and a t table.

    

    CI =  sample_mean - (t_star * standard_error), sample_mean + (t_star * standard_error)

    

    return f"Interpretation: With a sample mean of {round(sample_mean, 2)}, we can be {round(conf * 100, 2)}% confident that the true population mean lies between {round(CI[0], 2)} and {round(CI[1], 2)}."
t_confidence_interval_sample_mean(titanic, "Age", n = 10, conf = 0.95)
t_confidence_interval_sample_mean(titanic, "Age", n = 25, conf = 0.95)
t_confidence_interval_sample_mean(titanic, "Fare", n = 25, conf = 0.80)
grouped_survival = titanic.groupby("Survived").size()

plt.bar(grouped_survival.index.values, grouped_survival.values, tick_label = ["Died", "Survived"])

plt.title("Population Distribution of Titanic Survival")

plt.show()
def z_confidence_interval_sample_prop(titanic, column, n, conf, success_value):

    

    sample = r.sample(list(titanic[column]), n) #Getting our sample of size n

    

    success_counter = 0

    for j in sample: #This section gets a list of proportions occurances for each discrete value, in this case 0 and 1

        if j == success_value:

            success_counter += 1

    p = success_counter / len(sample) #We get our sample proportion

    

    if n * p * (1 - p) < 10:

        return "Normality not achieved.  Get a bigger sample size."

    

    standard_error = np.sqrt(p * (1 - p) / n)

    

    z_star = stats.norm.ppf(((1 - conf) / 2) + conf)

    

    CI =  p - (z_star * standard_error), p + (z_star * standard_error)

    

    return f"Interpretation: With a sample proportion of {round(p, 3)}, we can be {round(conf * 100, 3)}% confident that the true population proportion lies between {round(CI[0], 3)} and {round(CI[1], 3)}."
z_confidence_interval_sample_prop(titanic, "Survived", 20, 0.95, 1)
z_confidence_interval_sample_prop(titanic, "Survived", 50, 0.95, 1)
z_confidence_interval_sample_prop(titanic, "Survived", 50, 0.50, 1)
z_confidence_interval_sample_prop(titanic, "Pclass", 50, 0.50, 1)
z_confidence_interval_sample_prop(titanic, "Pclass", 75, 0.95, 1)