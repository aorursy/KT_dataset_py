import numpy as np

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import pandas as pd

from IPython import display



pd.options.display.max_rows = 10

plt.style.use("ggplot")

warnings.filterwarnings("ignore")
age = np.array([1,2,3,5,6,7,7,10,12,13])



# Mean

mean_age = np.mean(age)

print(mean_age)



# Median

median_age = np.median(age)

print(median_age)



# Mode

mode_age = stats.mode(age)

print(mode_age)
np.var(age) # Variance of the age array
np.std(age)  # Standart devitation of the age
y = np.random.uniform(5,8,100)

x1 = np.random.uniform(10,20,100)

x2 = np.random.uniform(0,30,100)

plt.scatter(x1,y,color="black")

plt.scatter(x2,y,color="orange")

plt.xlim([-1,31])

plt.ylim([2,11])

plt.xlabel("x")

plt.ylabel("y")

print("X1 mean: {} and meadian: {}".format(np.mean(x1),np.median(x1)))

print("X2 mean: {} and meadian: {}".format(np.mean(x2),np.median(x2)))
data = pd.read_csv("../input/data.csv")

data = data.drop(['Unnamed: 32','id'],axis = 1)

data.head()
benign = data[data["diagnosis"] == "B"]

malignant = data[data["diagnosis"] == "M"]

desc = benign.radius_mean.describe()

desc
Q1 = desc[4] # %25 value

Q3 = desc[6] # %75 value

IQR = Q3-Q1

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR

print("Anything outside this range is an outlier: [{:5.3f}, {:5.3f}]".format(lower_bound, upper_bound))
benign[benign.radius_mean < lower_bound].radius_mean
print("Outliers: ", benign[(benign.radius_mean < lower_bound) | (benign.radius_mean > upper_bound)].radius_mean.values)
melted_data = pd.melt(data, id_vars = "diagnosis", value_vars = ['radius_mean'])

#   | diagnosis | variable  | value 

# 0 |         M| radius_mean| 17.99



sns.boxplot(x='variable', y='value', data=melted_data, hue='diagnosis')

plt.show()
f, ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt='.1f', ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.show()
sns.jointplot(x='radius_mean', y='area_mean', data=data, kind='reg')

sns.jointplot(x='radius_mean', y='fractal_dimension_mean', data=data, kind='reg')

plt.show()
# Let's look at the relation with more than one data

df = data.loc[:,["radius_mean","area_mean","fractal_dimension_se"]]
g = sns.PairGrid(df, diag_sharey=False)

g.map_lower(sns.kdeplot, cmap='Blues_d') # kernel density plot

g.map_upper(sns.scatterplot)

g.map_diag(sns.kdeplot, lw=3)  # kernel density plot

plt.show()
# [cov(a,a)  cov(a,b)

# cov(a,b)  cov(b,b)]

# Numpy returns the result as 2x2 array. So select the diagonal ones.

np.cov(data.radius_mean, data.area_mean)[0][1]
print("Covariance between radius mean and area mean: ",data.radius_mean.cov(data.area_mean))

print("Covariance between radius mean and fractal dimension se: ",data.radius_mean.cov(data.fractal_dimension_se))
fig, (ax1, ax2) = plt.subplots(1, 2)

sns.scatterplot(data.radius_mean, data.area_mean, ax=ax1)

sns.scatterplot(data.fractal_dimension_se, data.radius_mean, ax=ax2)

plt.tight_layout()

plt.show()
p1 = data.loc[:,["area_mean","radius_mean"]].corr(method= "pearson")

p2 = np.cov( data.radius_mean, data.area_mean)/(data.radius_mean.std()*data.area_mean.std())
p1
p2
sns.jointplot(data.radius_mean, data.area_mean, kind="reg")

plt.show()
spear = data.loc[:,["area_mean","radius_mean"]].corr(method= "spearman")

spear
kendall = data.loc[:,["area_mean","radius_mean"]].corr(method= "kendall")

kendall
def cohend(d1, d2):

    # calculate the size of samples

    n1, n2 = len(d1), len(d2)

    # calculate the variance of the samples

    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)

    # calculate the pooled standard deviation

    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    # calculate the means of the samples

    u1, u2 = np.mean(d1), np.mean(d2)

    # calculate the effect size

    return (u1 - u2) / s
cohend(malignant.radius_mean, benign.radius_mean)
# dice example

a = np.random.randint(1,7,60000)

print("sample space: ",np.unique(a))

plt.hist(a, bins=12) # bins =12 for pretty plot. Normally it is 6

plt.ylabel("Number of outcomes")

plt.xlabel("Possible outcomes")

plt.show()
# dice rolling

n = 2 # number of trials

p = 0.5 # probability of each trial

s = np.random.binomial(n, p, size=10000) # 10000 = number of test

weights = np.ones_like(s)/float(len(s))

plt.hist(s, weights=weights)

plt.xlabel("number of success")

plt.ylabel("probability")

plt.show()
n = 10

r = 4 # success

p = 1/6 # success rate



stats.binom.pmf(r,n,p) # probability mass function
lamda = 3

s1 = np.random.poisson(lamda, size=100000)

weights1 = np.ones_like(s1)/float(len(s1))

plt.hist(s1, weights=weights1, bins = 100)

plt.xlabel("number of occurances") 

plt.ylabel("probability")
# parameters of normal distribution

mu, sigma = 110, 20  # mean and standard deviation

s = np.random.normal(mu, sigma, size=100000)

print("mean: ", np.mean(s))

print("standart deviation: ", np.std(s))



# visualize with histogram

plt.figure(figsize = (10,7))

plt.hist(s, 100, normed=False)

plt.ylabel("frequency")

plt.show()