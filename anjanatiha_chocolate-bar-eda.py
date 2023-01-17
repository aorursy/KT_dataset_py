# System

import os

import sys



# Numerical

import numpy as np

from numpy import median

import pandas as pd





# NLP

import re

from string import ascii_letters





# Tools

import itertools

import pycountry





# Machine Learning - Preprocessing

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler





# Machine Learning - Model Selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV





# Machine Learning - Models

from sklearn import svm

from sklearn.svm import SVC

from sklearn.svm import LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomTreesEmbedding, RandomForestClassifier, VotingClassifier

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB 

from sklearn.neighbors import KDTree, KNeighborsClassifier, NearestNeighbors

from sklearn.neural_network import BernoulliRBM, MLPClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.mixture import GaussianMixture





# Machine Learning - Evaluation

from sklearn import metrics 

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score

from sklearn.utils import class_weight





# Plot

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import seaborn as sns



print(os.listdir("../input"))
df_org = pd.read_csv("../input/flavors_of_cacao.csv")

df = pd.read_csv("../input/flavors_of_cacao.csv")

df.head()
df.columns
df["Cocoa Percent"] = df["Cocoa\nPercent"].apply(lambda x: (int(float(x[:-1]))//5)*5)
col = "Cocoa Percent"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col).title()

xlabel = title

ylabel = "Count"



figsize = (18, 6)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



ax = sns.distplot(df[col])

ax.set_title(title, fontsize=title_fontsize)

ax.set_xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

ax.set_ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

ax.tick_params(labelsize=xtick_fontsize)
col = "Rating"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col).title()

xlabel = title

ylabel = "Count"



figsize = (18, 8)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



sns.distplot(df[col])

ax = sns.distplot(df[col])

ax.set_title(title, fontsize=title_fontsize)

ax.set_xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

ax.set_ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

ax.tick_params(labelsize=xtick_fontsize)
col1 = "Cocoa Percent"

col2 = "Rating"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col1).title() + "   VS   " + re.sub("[^a-zA-Z0-9]", " ", col2).title()

xlabel = title

ylabel = ""



figsize = (18, 6)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



ax = sns.boxplot(x=col1, y=col2, data=df)

plt.title(title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize, rotation=90)

plt.yticks(fontsize=ytick_fontsize)
def get_city_country(text, region="all", count=1):

    subdivision_list= []

    country_list = []

    region_list = []

    

    if region=="subdivision" or region=="all":

        for subdivision in pycountry.subdivisions:

            if subdivision.name in text: 

                subdivision_list.append(subdivision.name)

                

    if region=="country" or region=="all":

        for country in pycountry.countries:

            if country.name in text: 

                country_list.append(country.name)

                

    if region=="subdivision": 

        return ", ".join(subdivision_list[:count])

    elif region=="country": 

        return ", ".join(country_list[:count])

    else: 

        for i in range(len(subdivision_list)):

            try:

                region_list.append(subdivision_list[i] +", "+ country_list[i])

            except: break

        return " | ".join(region_list[:count])

    
df["Specific Bean Origin Country"] = df["Specific Bean Origin\nor Bar Name"].apply(lambda x: get_city_country(x, "country", 1))

df["Specific Bean Origin Subdivision"] = df["Specific Bean Origin\nor Bar Name"].apply(lambda x: get_city_country(x, "subdivision", 1))

# df["Location"] = df["Specific Bean Origin\nor Bar Name"].apply(lambda x: get_city_country(x, "all", 1))
col = "Specific Bean Origin Country"

c = 30

title = col

xlabel = title

ylabel = "Count"



figsize = (18, 8)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)

d = df[df[col]!=""]

d[col].value_counts().sort_values(ascending=False).head(c).plot(kind = 'bar')

plt.title("Top " + str(c) + " " + title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize)

plt.yticks(fontsize=ytick_fontsize)
col = "Specific Bean Origin Subdivision"

c = 30

title = col

xlabel = title

ylabel = "Count"



figsize = (18, 8)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)

d = df[df[col]!=""]

d[col].value_counts().sort_values(ascending=False).head(c).plot(kind = 'bar')

plt.title("Top " + str(c) + " " + title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize)

plt.yticks(fontsize=ytick_fontsize)
col = "Company\nLocation"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col).title()

xlabel = title

ylabel = "Count"



figsize = (18, 8)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



df[col].value_counts().sort_values(ascending=False).head(c).plot(kind = 'bar')

plt.title("Top " + str(c) + " " + title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize)

plt.yticks(fontsize=ytick_fontsize)
col1 = "Specific Bean Origin Country"

col2 = "Rating"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col1).title() + "   VS   " + re.sub("[^a-zA-Z0-9]", " ", col2).title()

xlabel = title

ylabel = ""



figsize = (18, 6)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



ax = sns.boxplot(x=col1, y=col2, data=df)

plt.title(title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize, rotation=90)

plt.yticks(fontsize=ytick_fontsize)
col1 = "Specific Bean Origin Country"

col2 = "Cocoa Percent"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col1).title() + "   VS   " + re.sub("[^a-zA-Z0-9]", " ", col2).title()

xlabel = title

ylabel = ""



figsize = (18, 6)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



ax = sns.boxplot(x=col1, y=col2, data=df)

plt.title(title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize, rotation=90)

plt.yticks(fontsize=ytick_fontsize)
col1 = "Specific Bean Origin Subdivision"

col2 = "Rating"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col1).title() + "   VS   " + re.sub("[^a-zA-Z0-9]", " ", col2).title()

xlabel = title

ylabel = ""



figsize = (18, 6)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



ax = sns.boxplot(x=col1, y=col2, data=df)

plt.title(title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize, rotation=90)

plt.yticks(fontsize=ytick_fontsize)
col1 = "Specific Bean Origin Subdivision"

col2 = "Cocoa Percent"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col1).title() + "   VS   " + re.sub("[^a-zA-Z0-9]", " ", col2).title()

xlabel = title

ylabel = ""



figsize = (18, 6)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



ax = sns.boxplot(x=col1, y=col2, data=df)

plt.title(title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize, rotation=90)

plt.yticks(fontsize=ytick_fontsize)
col1 = "Company\nLocation"

col2 = "Rating"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col1).title() + "   VS   " + re.sub("[^a-zA-Z0-9]", " ", col2).title()

xlabel = title

ylabel = ""



figsize = (18, 6)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



ax = sns.boxplot(x=col1, y=col2, data=df)

plt.title(title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize, rotation=90)

plt.yticks(fontsize=ytick_fontsize)
col1 = "Company\nLocation"

col2 = "Cocoa Percent"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col1).title() + "   VS   " + re.sub("[^a-zA-Z0-9]", " ", col2).title()

xlabel = title

ylabel = ""



figsize = (18, 6)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



ax = sns.boxplot(x=col1, y=col2, data=df)

plt.title(title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize, rotation=90)

plt.yticks(fontsize=ytick_fontsize)
col1 = "Specific Bean Origin Country"

col2 = "Cocoa Percent"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col1).title() + "   VS   " + re.sub("[^a-zA-Z0-9]", " ", col2).title()

xlabel = title

ylabel = ""



figsize = (18, 6)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



ax = sns.boxplot(x=col1, y=col2, data=df)

plt.title(title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize, rotation=90)

plt.yticks(fontsize=ytick_fontsize)
col = "Bean\nType"

c = 8

title = re.sub("[^a-zA-Z0-9]", " ", col).title()

xlabel = title

ylabel = "Count"



figsize = (18, 8)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



df[col].value_counts().sort_values(ascending=False).head(c).plot(kind = 'bar')

plt.title("Top " + str(c) + " " + title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize)

plt.yticks(fontsize=ytick_fontsize)
col = "Broad Bean\nOrigin"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col).title()

xlabel = title

ylabel = "Count"



figsize = (18, 8)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



df[col].value_counts().sort_values(ascending=False).head(c).plot(kind = 'bar')

plt.title("Top " + str(c) + " " + title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize)

plt.yticks(fontsize=ytick_fontsize)
col1 = "Broad Bean\nOrigin"

col2 = "Rating"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col1).title() + "   VS   " + re.sub("[^a-zA-Z0-9]", " ", col2).title()

xlabel = title

ylabel = ""



figsize = (10, 30)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



ax = sns.boxplot(x=col2, y=col1, data=df)

plt.title(title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize, rotation=90)

plt.yticks(fontsize=ytick_fontsize)
col1 = "Broad Bean\nOrigin"

col2 = "Cocoa Percent"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col1).title() + "   VS   " + re.sub("[^a-zA-Z0-9]", " ", col2).title()

xlabel = title

ylabel = ""



figsize = (10, 30)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



ax = sns.boxplot(x=col2, y=col1, data=df)

plt.title(title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize, rotation=90)

plt.yticks(fontsize=ytick_fontsize)
plt.figure(figsize=(18, 8))



col = "Review\nDate"

title = re.sub("[^a-zA-Z0-9]", " ", col).title()

xlabel = title

ylabel = "Count"



figsize = (18, 8)



fontsize = 16

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



ax = sns.countplot(df[col])

ax.set_title(title, fontsize=title_fontsize)

ax.set_xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

ax.set_ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

ax.tick_params(labelsize=xtick_fontsize)
col1 = "Review\nDate"

col2 = "Rating"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col1).title() + "   VS   " + re.sub("[^a-zA-Z0-9]", " ", col2).title()

xlabel = title

ylabel = ""



figsize = (18, 6)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



ax = sns.boxplot(x=col1, y=col2, data=df)

plt.title(title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize)

plt.yticks(fontsize=ytick_fontsize)
col1 = "Review\nDate"

col2 = "Cocoa Percent"

c = 10

title = re.sub("[^a-zA-Z0-9]", " ", col1).title() + "   VS   " + re.sub("[^a-zA-Z0-9]", " ", col2).title()

xlabel = title

ylabel = ""



figsize = (18, 6)



fontsize = 14

title_fontsize = fontsize*2

xlabel_fontsize = fontsize*1.3

ylabel_fontsize = fontsize*1.3

xtick_fontsize = fontsize

ytick_fontsize = fontsize





plt.figure(figsize=figsize)



ax = sns.boxplot(x=col1, y=col2, data=df)

plt.title(title, fontsize=title_fontsize)

plt.xlabel(xlabel=xlabel, fontsize=xlabel_fontsize)

plt.ylabel(ylabel=ylabel, fontsize=ylabel_fontsize)

plt.xticks(fontsize=xtick_fontsize, rotation=90)

plt.yticks(fontsize=ytick_fontsize)
cols = df.columns

cols = ['Rating', 'Bean\nType', 'Cocoa Percent', 'Specific Bean Origin Country', 'Specific Bean Origin Subdivision']

cols