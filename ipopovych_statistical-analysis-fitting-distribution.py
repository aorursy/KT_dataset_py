import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy

%matplotlib inline

import warnings; warnings.simplefilter('ignore')
DATA = pd.read_csv("../input/OpenData2018.csv", sep=";")
DATA.head()
grades = DATA[["UkrBall100", "mathBall100", "engBall100", "physBall100", "geoBall100", "histBall100",

               'chemBall100', 'bioBall100', 'fraBall100',  'deuBall100',  'spaBall100']].replace({0:np.nan})
grades.describe()
dat = grades.histBall100 

st = (dat.mean() - dat.min()) / dat.std() # calculate the statistics



print("Test Statistics: ", st)
def plot_distr(subject):

    col = subject+"Ball100"

    dat = DATA[col].replace({0:np.nan}).dropna()

    sns.distplot(dat, bins=100, color="c")

    

plot_distr("hist") # History
def fit_distribution(data):

    distributions = ["expon", "weibull_max", "pareto", "genextreme"]

    dist_results = []

    params = {}

    

    plt.hist(data, normed=True)

    rX = np.linspace(100,200, 100)

    

    for dist_name in distributions:

        # fit the distribution

        dist = getattr(scipy.stats, dist_name)

        param = dist.fit(data)

        params[dist_name] = param

        

        # Use the Kolmogorov-Smirnov test

        D, p = scipy.stats.kstest(data, dist_name, args=param)



        dist_results.append((dist_name, p, D))

        

        rP = dist.pdf(rX, *param)

        plt.plot(rX, rP, label=dist_name)

        

        print(dist_name.ljust(16) + ("p-value: "+str(p)).ljust(16) + "D: "+str(D))    

    plt.legend()

    plt.show()

            

    return params, dist_results





params, results = fit_distribution(list(grades.histBall100.dropna()))
# find minimum distance between the CDF's of the two samples

best_dist, best_p, best_d = (min(results, key=lambda item: item[2]))

    

print("Most close fitting distribution: "+str(best_dist))

print("It's p-value: "+ str(best_p))

print("Parameters for the most close fit: "+ str(params[best_dist]))
regdata = DATA[["REGNAME", "UkrBall100", "mathBall100", "engBall100", "physBall100", "geoBall100", "histBall100",

               'chemBall100', 'bioBall100', 'fraBall100',  'deuBall100',  'spaBall100']].replace({0:np.nan})



def plot_region_avg(sub):

    col1 = "REGNAME"

    col2 = sub +"Ball100"

    ss = regdata[[col1, col2]].dropna() # select subject and clean data

    b1, b2 = [], []

    

    for region in sorted(set(ss[col1])):

        people = ss[ss[col1] == region]

        mean = people[col2].mean()

        b1.append(region)

        b2.append(mean)

        #print(region, mean)

    regData = pd.DataFrame( {col1: b1, sub: b2})

    regplot = sns.barplot(x="REGNAME", y=sub, data=regData)

    regplot.set_xticklabels(regplot.get_xticklabels(), rotation=90)

    plt.show()



plot_region_avg("hist")
plot_region_avg("math")
plot_region_avg("phys")
plot_region_avg("Ukr")
sns.heatmap(grades.corr())

plt.show()
# physics and mathematics



df = DATA[np.isfinite(DATA['physBall100'])]

df = df[np.isfinite(df['mathBall100'])]

scipy.stats.linregress(df.mathBall100, df.physBall100)
def lin(sub1, sub2):

    col1 = sub1+"Ball100"

    col2 = sub2+"Ball100"

    ss = DATA[[col1, col2]].replace({0:np.nan}).dropna() #select two subjects and clean data

    b1, b2 = [], []

    for ball in sorted(set(ss[col1])):

        people = ss[ss[col1] == ball]

        mean = people[col2].mean()

        b1.append(ball)

        b2.append(mean)

    return pd.DataFrame( {sub1: b1, sub2: b2})



s1, s2 = "phys", "math"

df = lin(s1, s2)

sns.lmplot(s1, s2, df)

plt.show()
# geography and spanish



df = DATA[np.isfinite(DATA['spaBall100'])]

df = df[np.isfinite(df['geoBall100'])]

scipy.stats.linregress(df.spaBall100, df.geoBall100)
s1, s2 = "geo", "spa"

df = lin(s1, s2)

sns.lmplot(s1, s2, df)

plt.show()
df = DATA.replace({0:np.nan})



def plot_subject(subject):

    count = ["SEXTYPENAME", "ClassProfileNAME", "ClassLangName"]

    plt.figure(figsize=(18,8))

    ax = sns.countplot(x=subject+'Ball100', hue=count[0], data=df, orient="h")

    #ax = sns.countplot(x="SEXTYPENAME", hue=subject+'TestStatus', data=DATA, orient="h")

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)

    plt.tight_layout()

    plt.show()

    

plot_subject("math")

plot_subject("hist")
plot_subject("Ukr")
plot_subject("phys")