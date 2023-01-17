# System

import os



# Numerical

import numpy as np

import pandas as pd



# NLP

import re



# Tools

import itertools



# Machine Learning - Preprocessing

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



# Machine Learning - Model Selection

from sklearn.model_selection import GridSearchCV



# Machine Learning - Models

from sklearn import svm



# Machine Learning - Evaluation

from sklearn import metrics 

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score



# Plot

import matplotlib.pyplot as plt

import seaborn as sns



print(os.listdir("../input"))
df = pd.read_csv("../input/employee_reviews.csv")
df.head()
df.describe()
# df.info()
all_cols = ['company', 'location', 'job-title', 'overall-ratings',

       'work-balance-stars', 'culture-values-stars',

       'carrer-opportunities-stars', 'comp-benefit-stars',

       'senior-mangemnet-stars', 'helpful-count']



rating_cols = ["overall-ratings", "work-balance-stars", "culture-values-stars",

       "carrer-opportunities-stars", "comp-benefit-stars",

       "senior-mangemnet-stars"]
sns.reset_defaults()



figsize=(20, 5)

ticksize = 14

titlesize = ticksize + 8

labelsize = ticksize + 5



sns.set(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=2, color_codes=True, rc=None)

sns.set_style("ticks", {"xtick.major.size": ticksize, "ytick.major.size": ticksize})







params = {'figure.figsize' : figsize,

          'axes.labelsize' : labelsize,

          'axes.titlesize' : titlesize}



plt.rcParams.update(params)





col = "company"



xlabel = "Company"

ylabel = "Count"



title = "Vote Count Per Company"





sns.countplot(x=df[col], data=df)

plt.title(title)

plt.xlabel(xlabel)

plt.ylabel(ylabel)

plt.plot()
df[rating_cols] = df[rating_cols].apply(pd.to_numeric, errors='coerce')
sns.reset_defaults()



figsize=(20, 16)

ticksize = 14

titlesize = ticksize + 8

labelsize = ticksize + 5



nrows = 3

ncols = 2



sns.set(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)

sns.set_style("ticks", {"xtick.major.size": ticksize, "ytick.major.size": ticksize})





fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

plt.subplots_adjust(hspace=.6, wspace=.3)



params = {'figure.figsize' : figsize,

          'axes.labelsize' : labelsize,

          'axes.titlesize' : titlesize}



plt.rcParams.update(params)





xcol = "company"

xlabel = "Company"



title = "Vote Count Per Company"





feature_count = len(rating_cols)



for i in range(feature_count):

    plt.subplot(nrows,ncols, i+1)

    ylabel = re.sub("[^a-zA-Z]", " ", rating_cols[i])

    ylabel = re.sub("\s+", " ", ylabel).title()

    tmp = df.groupby(xcol, as_index=False)[rating_cols[i]].mean()

    sns.barplot(x=xcol, y=rating_cols[i], data=tmp)

    plt.title(ylabel)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.plot()



plt.show()
def get_state(x):

    if "(" in x:

        x = x.split("(")[1]

        x = x.split(")")[0]

    elif ", " in x:

        x = x.split(", ")[-1]



    return x



df["state"] = df["location"].apply(lambda x: get_state(x))
a = get_state("Chinno Hills, CA")

a
sns.reset_defaults()



figsize=(30, 50)

ticksize = 14

titlesize = ticksize + 8

labelsize = ticksize + 5



nrows = len(rating_cols)

ncols = 1





sns.set(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=2, color_codes=True, rc=None)

sns.set_style("ticks", {"xtick.major.size": ticksize, "ytick.major.size": ticksize})





fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

plt.subplots_adjust(hspace=2, wspace=.3)



params = {'figure.figsize' : figsize,

          'axes.labelsize' : labelsize,

          'axes.titlesize' : titlesize}



plt.rcParams.update(params)





feature_count = len(rating_cols)

item_count = 50

xcol = "state"

xlabel = "State"





for i in range(feature_count):

    plt.subplot(nrows,ncols, i+1)

    ylabel = re.sub("[^a-zA-Z]", " ", rating_cols[i])

    ylabel = re.sub("\s+", " ", ylabel).title()

    tmp = df.groupby(xcol, as_index=False)[rating_cols[i]].mean().sort_values(by=rating_cols[i],ascending=False)

    sns.barplot(x=xcol, y=rating_cols[i], data=tmp.head(item_count))

    plt.title(ylabel)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.xticks(rotation=90)

    plt.plot()



plt.show()

sns.reset_defaults()



figsize=(30, 50)

ticksize = 14

titlesize = ticksize + 8

labelsize = ticksize + 5



nrows = len(rating_cols)

ncols = 1



sns.set(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=2, color_codes=True, rc=None)

sns.set_style("ticks", {"xtick.major.size": ticksize, "ytick.major.size": ticksize})





fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

plt.subplots_adjust(hspace=2, wspace=.3)



params = {'figure.figsize' : figsize,

          'axes.labelsize' : labelsize,

          'axes.titlesize' : titlesize}



plt.rcParams.update(params)





feature_count = len(rating_cols)

item_count = 50

xcol = "location"

xlabel = "Location"





for i in range(feature_count):

    plt.subplot(nrows,ncols, i+1)

    ylabel = re.sub("[^a-zA-Z]", " ", rating_cols[i])

    ylabel = re.sub("\s+", " ", ylabel).title()

    tmp = df.groupby(xcol, as_index=False)[rating_cols[i]].mean().sort_values(by=rating_cols[i],ascending=False)

    sns.barplot(x=xcol, y=rating_cols[i], data=tmp.head(item_count))

    plt.title(ylabel)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.xticks(rotation=90)

    plt.plot()



plt.show()

sns.reset_defaults()



figsize = (16, 6)

ticksize = 14

titlesize = ticksize + 8

labelsize = ticksize + 5



sns.set(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=2, color_codes=True, rc=None)

sns.set_style("ticks", {"xtick.major.size": ticksize, "ytick.major.size": ticksize})





params = {'figure.figsize' : figsize,

          'axes.labelsize' : labelsize,

          'axes.titlesize' : titlesize}



plt.rcParams.update(params)





xcol = "company"

ycol = "helpful-count"



xlabel = "Company"

ylabel = "Helpful Count"



title = "Vote Count Per Company"





tmp = df.groupby(xcol, as_index=False)[ycol].count()

sns.barplot(x=xcol, y=ycol, data=tmp)

plt.title(ylabel)

plt.xlabel(xlabel)

plt.ylabel(ylabel)

plt.show()

tmp = df.groupby(xcol, as_index=False)[ycol].mean()

sns.barplot(x=xcol, y=ycol, data=tmp)

plt.title(ylabel)

plt.xlabel(xlabel)

plt.ylabel(ylabel)

plt.plot()

plt.show()
