import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16,8)
df_ath = pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")
print(df_ath.shape)
df_ath.head()
df_ath.isnull().sum()
features = ["Age","Height","Weight","Year"]
df_ath.loc[:,features].describe()
plt.figure(figsize=(16,4))
for i,f in enumerate(features):
    plt.subplot(1,len(features),i+1)
    plt.boxplot(df_ath[f].dropna(), patch_artist=True, showfliers=True)
    plt.xticks([], [])
    plt.title(f)
plt.show()
df_ath.loc[df_ath.Age == max(df_ath.Age)]
df_ath.loc[df_ath.Height == max(df_ath.Height)]
seasons = ["Summer","Winter"]
n_sports = 10
plt.figure(figsize=(16,10))
for i,s in enumerate(seasons):
    df = df_ath.loc[df_ath.Games.str.contains(s)]
    counts = df.groupby("Sport").size().sort_values(ascending=False)[:n_sports]
    plt.subplot(2,1,i+1)
    counts.plot.bar(rot=0)
    plt.ylabel("Number of Entries")
    plt.title(s)
n_sports = 8
plt.figure(figsize=(16,10))
young = df_ath.loc[df_ath.Age < 15].groupby("Sport").size().sort_values(ascending=False)[:n_sports]
old = df_ath.loc[df_ath.Age > 55].groupby("Sport").size().sort_values(ascending=False)[:n_sports]
data = [young,old]
labels = ["Younger than 15","Older than 55"]
for i,series in enumerate(data):
    plt.subplot(2,1,i+1)
    series.plot.bar(rot = 0)
    plt.title(labels[i])
    plt.ylabel("Number of Entries")
plt.show()
# import country names 
df_countries = pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv")
countries = {series.NOC : series.region for (_,series) in df_countries.iterrows()}
# fix bug in countries data
countries["SGP"] = countries["SIN"]
n_countries = 15
s = df_ath.groupby("NOC").size().sort_values(ascending=False)[:n_countries]
s.index = [countries[i] for i in s.index]
plt.figure(figsize=(16,4))
s.plot.bar(rot=0)
plt.ylabel("Number of Entries")
plt.title("Total Entries Over History of Dominant Countries")
plt.show()
len(df_ath.loc[df_ath.Year == 1896].loc[df_ath.Sex=="F"])
# define a function to calculate the sex ratio
sexratio = lambda df : np.sum(df.Sex=="F")/df.shape[0]
# group the data by sport and year 
grouped = df_ath.groupby(["Sport","Year"])
for sport in df_ath.Sport.unique():
    years = sorted(df_ath.loc[df_ath.Sport==sport].Year.unique())
    ratios = [sexratio(grouped.get_group((sport,y))) for y in years]
    # print sport if women dominated for at least 3 years in history
    if np.sum(np.array(ratios) > 0.5) >= 3:
        print(sport)
all_sports = df_ath.Sport.unique()
print("Number of sports: %i" % len(all_sports))
print()
print(all_sports)
# group the dataset by sport, year and then sex 
grouped = df_ath.sort_values(by="Year").groupby(["Sport","Year"])
def expWeighted(data, beta = 0.8):
    """ Computes the exponentially weighted average of time series data. beta controls the weighting on history, where beta = 0 reduces to the time series itself."""
    v = data[0]
    result = np.zeros(len(data))
    for i in range(len(data)):
        v = beta*v + (1-beta)*data[i] 
        result[i] = v
    return result
beta = 0.5 # exponential weighting parameter, 0 = no weighting on history
sports = ["Athletics","Swimming","Cycling","Boxing","Basketball","Trampolining","Figure Skating"] # sports under consideration 
plt.figure(figsize=(15,8))
# compute the ratio over time for each sport 
for s in sports:
    years = sorted(df_ath.Year[df_ath.Sport == s].unique())
    sex_ratios = np.array([sexratio(grouped.get_group((s,y))) for y in years])
    plt.plot(years, expWeighted(sex_ratios, beta=beta), linewidth=2)
# compute the ratio over time over all sports
years = sorted(df_ath.Year.unique())
sex_ratios = [sexratio(df_ath.loc[df_ath.Year==y]) for y in years]
plt.plot(years, expWeighted(sex_ratios, beta=beta), c = "Grey", linestyle = "--",linewidth=3)
plt.plot([min(years),max(years)],[0.5,0.5],c="k",linestyle="--")
plt.title("Fraction of female Olympic participants by year")
plt.xlabel("Year")
plt.ylabel("Fraction of female participants")
plt.legend(sports + ["All Sports"])
plt.show()     
df_continents = pd.read_csv("../input/world-countries-and-continents/countries and continents.csv") 
# continent by official olympic committee
continents = {series.IOC : series.Continent for (_,series) in df_continents.iterrows()}
# continent by country name 
cont_by_country = {series.official_name_en : series.Continent for (_,series) in df_continents.iterrows()}
african = []
for noc in countries.keys():
    if noc in continents:
        if continents[noc]=="AF":
            african.append(noc) 
    elif countries[noc] in cont_by_country:
        if cont_by_country[countries[noc]]=="AF":
            african.append(noc) 
    else:
        print(noc, countries[noc])
nations = ["FRA","GBR","USA"]
years = df_ath.loc[df_ath.Year > 1920].loc[df_ath.Games.str.contains("Summer")].Year.sort_values().unique()
grouped = df_ath.groupby(df_ath.Year)
getRatio = lambda df, nocs : np.sum(df.NOC.isin(nocs))/df.shape[0]

plt.figure(figsize=(15,5))
ratios = np.array([getRatio(grouped.get_group(y), african) for y in years])
plt.plot(years,expWeighted(ratios*100))
for noc in nations:
    ratios = np.array([getRatio(grouped.get_group(y), [noc]) for y in years])
    plt.plot(years,expWeighted(ratios*100))
    
plt.legend(["African"] + [countries[noc] for noc in nations])
plt.ylabel("Percentage of Entries")
plt.xlabel("Summer games year")
plt.show()
sport = "Swimming"
sex = "F"
df_sport = df_ath.loc[df_ath.Sport == sport].loc[df_ath.Sex == sex]
# group by medal / no medal
grouped = df_sport.groupby(df_sport.Medal.notnull())
# function to normalize histograms 
getWeights = lambda x : (1/len(x))*np.ones_like(x)
plt.figure(figsize=(14,10))
for i,attr in enumerate(["Height","Weight","Age"]):
    medal    = grouped.get_group(True)[attr].dropna()
    no_medal = grouped.get_group(False)[attr].dropna()
    plt.subplot(2,2,i+1)
    plt.hist([medal, no_medal], weights = [getWeights(medal), getWeights(no_medal)])
    plt.legend(["Medal", "No Medal"])
    plt.title("%s dists of medal vs no-medal" % attr)
plt.show()
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
X = df_sport.loc[:,["Height","Weight","Age"]].dropna()
y = df_sport.loc[X.index,:].Medal.notnull()
clf = GaussianNB()
scores = cross_val_score(clf, X, y, cv =5)
print(scores)
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer

cross_val_score(clf, X, y, scoring=make_scorer(recall_score), cv=5)
clf.fit(X,y)
ypred = clf.predict(X)
L = len(ypred)
plt.figure(figsize=(16,3))
plt.subplot(1,2,1)
plt.title("Model")
plt.bar(["Medal","No Medal"],[np.sum(ypred)/L,np.sum(ypred == False)/L])
plt.subplot(1,2,2)
plt.title("True values")
plt.bar(["Medal","No Medal"],[np.sum(y)/L,np.sum(y == False)/L])
plt.show()
print("Model predicts %i out of %i instances are medalists" % (np.sum(ypred), len(ypred)))
from sklearn.metrics import precision_score
cross_val_score(clf, X, y, scoring=make_scorer(precision_score), cv=3)
from sklearn.neighbors import KernelDensity

df = df_sport.loc[df_sport.Height.notnull() & df_sport.Weight.notnull()]
features = ["Height","Weight"]
units = ["cm","kg"]
kde = KernelDensity(bandwidth=5)

plt.figure(figsize=(16,3))
for i, feat in enumerate(features):
    # weight / height data 
    x = df[feat]
    x_range = np.linspace(min(x),max(x),1000)
    # fit kernel density to data
    kde.fit(x[:,None])
    # compute the log probability over a range 
    logprob = kde.score_samples(x_range[:, None])
    plt.subplot(1,2,i+1)
    plt.fill_between(x_range, np.exp(logprob))
    plt.xlabel(feat + " in " + units[i])
    plt.ylabel("Probability Density")
plt.show()
plt.figure(figsize=(15,4))
labels = ["No Medal","Medal"]
h_range = np.linspace(155,190,50)
w_range = np.linspace(40,80,50)
X,Y = np.meshgrid(h_range,w_range)
# plot 2D probability density functions for medalists and non-medalists  
for i, medal in enumerate([False,True]):
    # 2D weight / height data
    D = df.loc[df.Medal.notnull() == medal].loc[:,["Height","Weight"]] 
    kde = KernelDensity(bandwidth=5)
    kde.fit(D)
    def F(x,y):
        return np.exp(kde.score_samples(np.array([[x,y]])))
    Z = np.vectorize(F)(X,Y)
    plt.subplot(1,2,i+1)
    cntr = plt.contour(h_range,w_range,Z)#,levels=[0.0,0.001,0.002])
    plt.clabel(cntr, inline=1, fontsize=10)
    plt.pcolor(h_range,w_range,Z)
    plt.colorbar()
    plt.xlabel("Height in cm")
    plt.ylabel("Weight in kg")
    plt.title(labels[i])
plt.show()