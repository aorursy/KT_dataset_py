# General

import numpy as np

import pandas as pd

import nltk

import random

import os

from os import path

from PIL import Image



# Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS



# Set Plot Theme

sns.set_palette([

    "#30a2da",

    "#fc4f30",

    "#e5ae38",

    "#6d904f",

    "#8b8b8b",

])

# Alternate # plt.style.use('fivethirtyeight')



# Pre-Processing

import string

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

import re

from nltk.stem import PorterStemmer



# Modeling

import statsmodels.api as sm

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.sentiment.util import *

from nltk.util import ngrams

from collections import Counter

from gensim.models import word2vec



# Warnings

import warnings

warnings.filterwarnings('ignore')
# Read and Peak at Data

df = pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv")

df.drop(df.columns[0],inplace=True, axis=1)



# Delete missing observations for following variables

for x in ["Division Name","Department Name","Class Name","Review Text"]:

    df = df[df[x].notnull()]



# Extracting Missing Count and Unique Count by Column

unique_count = []

for x in df.columns:

    unique_count.append([x,len(df[x].unique()),df[x].isnull().sum()])



# Missing Values

print("Missing Values: {}".format(df.isnull().sum().sum()))



# Data Dimensions

print("Dataframe Dimension: {} Rows, {} Columns".format(*df.shape))



# Create New Variables: 

# Word Length

df["Word Count"] = df['Review Text'].str.split().apply(len)

# Character Length

df["Character Count"] = df['Review Text'].apply(len)

# Boolean for Positive and Negative Reviews

df["Label"] = 0

df.loc[df.Rating >= 3,["Label"]] = 1
df.sample(3)
print("Dataframe Dimension: {} Rows, {} Columns".format(*df.shape))

pd.DataFrame(unique_count, columns=["Column","Unique","Missing"]).set_index("Column").T
df.describe().T.drop("count",axis=1)
df[["Title", "Division Name","Department Name","Class Name"]].describe(include=["O"]).T.drop("count",axis=1)
# Continous Distributions

f, ax = plt.subplots(1,3,figsize=(12,4), sharey=False)

sns.distplot(df.Age, ax=ax[0])

ax[0].set_title("Age Distribution")

ax[0].set_ylabel("Density")

sns.distplot(df["Positive Feedback Count"], ax=ax[1])

ax[1].set_title("Positive Feedback Count Distribution")

sns.distplot(np.log10((df["Positive Feedback Count"][df["Positive Feedback Count"].notnull()]+1)), ax=ax[2])

ax[2].set_title("Positive Feedback Count Distribution\n[Log 10]")

ax[2].set_xlabel("Log Positive Feedback Count")

plt.tight_layout()

plt.show()
# Percentage Accumulation from "Most Wealthy"

def percentage_accumulation(series, percentage):

    return (series.sort_values(ascending=False)

            [:round(series.shape[0]*(percentage/100))]

     .sum()/series

     .sum()*100)



# Gini Coefficient- Inequality Score

# Source: https://planspace.org/2013/06/21/how-to-calculate-gini-coefficient-from-raw-data-in-python/

def gini(list_of_values):

    sorted_list = sorted(list_of_values)

    height, area = 0, 0

    for value in sorted_list:

        height += value

        area += height - value / 2.

    fair_area = height * len(list_of_values) / 2.

    return (fair_area - area) / fair_area



# Cumulative Percentage of Positive Feedback assigned Percent of Reviewers (from most wealthy)

inequality = []

for x in list(range(100)):

    inequality.append(percentage_accumulation(df["Positive Feedback Count"], x))



# Generic Matplotlib Plot

plt.plot(inequality)

plt.title("Percentage of Positive Feedback by Percentage of Reviews")

plt.xlabel("Review Percentile starting with Feedback")

plt.ylabel("Percent of Positive Feedback Received")

plt.axvline(x=20, c = "r")

plt.axvline(x=53, c = "g")

plt.axhline(y=78, c = "y")

plt.axhline(y=100, c = "b", alpha=.3)

plt.show()



# 80-20 Rule Confirmation

print("{}% of Positive Feedback belongs to the top 20% of Reviews".format(

    round(percentage_accumulation(df["Positive Feedback Count"], 20))))



# Gini

print("\nGini Coefficient: {}".format(round(gini(df["Positive Feedback Count"]),2)))
# Cumulative Percentage of Positive Feedback assigned Percent of Reviewers (from most wealthy)

top_20 = df["Positive Feedback Count"].sort_values(ascending=False)[:round(df.shape[0]*(20/100))]



inequality = []

for x in list(range(100)):

    inequality.append(percentage_accumulation(top_20, x))



# Generic Matplotlib Plot

plt.plot(inequality)

plt.title("Percentage of Positive Feedback by Percentage of Reviews")

plt.xlabel("Review Percentile starting with Feedback")

plt.ylabel("Percent of Positive Feedback Received")

plt.axvline(x=20, c = "r")

plt.axhline(y=47, c = "r")

plt.axhline(y=100, c = "b", alpha=.3)



plt.show()



# 80-20 Rule Confirmation

print("{}% of Positive Feedback belongs to the top 20% of Reviews".format(

    round(percentage_accumulation(top_20, 20))))



# Gini

print("\nGini Coefficient: {}".format(round(gini(top_20),2)))
row_plots = ["Division Name","Department Name"]

f, axes = plt.subplots(1,len(row_plots), figsize=(14,4), sharex=False)



for i,x in enumerate(row_plots):

    sns.countplot(y=x, data=df,order=df[x].value_counts().index, ax=axes[i])

    axes[i].set_title("Count of Categories in {}".format(x))

    axes[i].set_xlabel("")

    axes[i].set_xlabel("Frequency Count")

axes[0].set_ylabel("Category")

axes[1].set_ylabel("")

plt.show()
# Clothing ID Category

f, axes = plt.subplots(1,2, figsize=[14,7])

num = 30

sns.countplot(y="Clothing ID", data = df[df["Clothing ID"].isin(df["Clothing ID"].value_counts()[:num].index)],

              order= df["Clothing ID"].value_counts()[:num].index, ax=axes[0])

axes[0].set_title("Frequency Count of Clothing ID\nTop 30")

axes[0].set_xlabel("Count")



sns.countplot(y="Clothing ID", data = df[df["Clothing ID"].isin(df["Clothing ID"].value_counts()[num:60].index)],

              order= df["Clothing ID"].value_counts()[num:60].index, ax=axes[1])

axes[1].set_title("Frequency Count of Clothing ID\nTop 30 to 60")

axes[1].set_ylabel("")

axes[1].set_xlabel("Count")

plt.show()



print("Dataframe Dimension: {} Rows".format(df.shape[0]))

df[df["Clothing ID"].isin([1078, 862,1094])].describe().T.drop("count",axis=1)
df.loc[df["Clothing ID"].isin([1078, 862,1094]),

       ["Title", "Division Name","Department Name","Class Name"]].describe(include=["O"]).T.drop("count",axis=1)
# Class Name

plt.subplots(figsize=(9,5))

sns.countplot(y="Class Name", data=df,order=df["Class Name"].value_counts().index)

plt.title("Frequency Count of Class Name")

plt.xlabel("Count")

plt.show()
#cat_dtypes = [x for x,y,z in unique_count if y < 10 and x not in ["Division Name","Department Name"]]

cat_dtypes = ["Rating","Recommended IND","Label"]

increment = 0

f, axes = plt.subplots(1,len(cat_dtypes), figsize=(14,4), sharex=False)



for i in range(len(cat_dtypes)):

    sns.countplot(x=cat_dtypes[increment], data=df,order=df[cat_dtypes[increment]].value_counts().index, ax=axes[i])

    axes[i].set_title("Frequency Distribution for\n{}".format(cat_dtypes[increment]))

    axes[i].set_ylabel("Occurrence")

    axes[i].set_xlabel("{}".format(cat_dtypes[increment]))

    increment += 1

axes[1].set_ylabel("")

axes[2].set_ylabel("")

plt.show()
f, axes = plt.subplots(2,4, figsize=(17,8), sharex=False)

for ii, xvar in enumerate(['Word Count', "Character Count"]):

    for i,y in enumerate(["Rating","Department Name","Recommended IND"]):

        for x in set(df[y][df[y].notnull()]):

            sns.kdeplot(df[xvar][df[y]==x], label=x, shade=False, ax=axes[ii,i])

        if ii is 0:

            axes[ii,i].set_title('{} Distribution (X)\nby {}'.format(xvar, y))

        else:

            axes[ii,i].set_title('For {} (X)'.format(xvar))

    axes[ii,0].set_ylabel('Occurrence Density')

    axes[ii,i].set_xlabel('')

    # Plot 4

    sns.kdeplot(df[xvar],shade=True,ax=axes[ii,3])

    axes[ii,3].set_xlabel("")

    if ii is 0:

        axes[ii,3].set_title('{} Distribution (X)\n'.format(xvar))

    else:

        axes[ii,3].set_title('For {} (X)'.format(xvar))

    axes[ii,3].legend_.remove()

plt.show()



print("Correlation Coefficient of Word Cound and Character Count: {}".format(

    round(df["Word Count"].corr(df["Character Count"]), 2)))



print("\nTotal Word Count is: {}".format(df["Word Count"].sum()))

print("Total Character Count is: {}".format(df["Character Count"].sum()))

df[["Word Count","Character Count"]].describe().T
# Heatmaps of Percentage Pivot Table

f, ax = plt.subplots(1,2,figsize=(16, 4), sharey=True)

sns.heatmap(pd.crosstab(df['Division Name'], df["Department Name"]),

            annot=True, linewidths=.5, ax = ax[0],fmt='g', cmap="Greens",

                cbar_kws={'label': 'Count'})

ax[0].set_title('Division Name Count by Department Name - Crosstab\nHeatmap Overall Count Distribution')



sns.heatmap(pd.crosstab(df['Division Name'], df["Department Name"], normalize=True).mul(100).round(0),

            annot=True, linewidths=.5, ax=ax[1],fmt='g', cmap="Greens",

                cbar_kws={'label': 'Percentage %'})

ax[1].set_title('Division Name Count by Department Name - Crosstab\nHeatmap Overall Percentage Distribution')

ax[1].set_ylabel('')

plt.tight_layout(pad=0)

plt.show()
# Heatmaps of Percentage Pivot Table

f, ax = plt.subplots(1,2,figsize=(16, 4), sharey=True)

sns.heatmap(pd.crosstab(df['Division Name'], df["Department Name"], normalize='columns').mul(100).round(0),

            annot=True, linewidths=.5, ax=ax[0],fmt='g', cmap="Greens",

                cbar_kws={'label': 'Percentage %'})

ax[0].set_title('Division Name Count by Department Name - Crosstab\nHeatmap % Distribution by Columns')



sns.heatmap(pd.crosstab(df['Division Name'], df["Department Name"], normalize='index').mul(100).round(0),

            annot=True, linewidths=.5, ax=ax[1],fmt='g', cmap="Greens",

                cbar_kws={'label': 'Percentage %'})

ax[1].set_title('Division Name Count by Department Name - Crosstab\nHeatmap % Distribution by Index')

ax[1].set_ylabel('')

plt.tight_layout(pad=0)

plt.show()
# Heatmaps of Percentage Pivot Table

f, ax = plt.subplots(1,2,figsize=(10, 7), sharey=True)

fsize = 13

sns.heatmap(pd.crosstab(df['Class Name'], df["Department Name"]),

            annot=True, linewidths=.5, ax = ax[0],fmt='g', cmap="Blues",

                cbar_kws={'label': 'Count'})

ax[0].set_title('Class Name Count by Department Name - Crosstab\nHeatmap Overall Count Distribution')



sns.heatmap(pd.crosstab(df['Class Name'], df["Department Name"], normalize=True).mul(100).round(0),

            annot=True, linewidths=.5, ax=ax[1],fmt='g', cmap="Blues",

                cbar_kws={'label': 'Percentage %'})

ax[1].set_title('Class Name Count by Department Name - Crosstab\nHeatmap Overall Percentage Distribution')

ax[1].set_ylabel('')

plt.tight_layout(pad=0)

plt.show()
# Heatmaps of Percentage Pivot Table

f, ax = plt.subplots(1,2,figsize=(10, 7), sharey=True)

fsize = 13

sns.heatmap(pd.crosstab(df['Class Name'], df["Department Name"], normalize = 'columns').mul(100).round(0)

            ,annot=True, fmt="g", linewidths=.5, ax=ax[0],cbar=False,cmap="Blues")

ax[0].set_title('Class Name Count by Count - Crosstab\nHeatmap % Distribution by Column', fontsize = fsize)

ax[1] = sns.heatmap(pd.crosstab(df['Class Name'], df["Department Name"], normalize = 'index').mul(100).round(0)

            ,annot=True, fmt="2g", linewidths=.5, ax=ax[1],cmap="Blues",

                cbar_kws={'label': 'Percentage %'})

ax[1].set_title('Class Name Count by Count - Crosstab\nHeatmap % Distribution by Index', fontsize = fsize)

ax[1].set_ylabel('')

plt.tight_layout(pad=0)

plt.show()
# Heatmaps of Percentage Pivot Table

f, ax = plt.subplots(1,2,figsize=(10, 7), sharey=True)

fsize = 13

sns.heatmap(pd.crosstab(df['Class Name'], df["Division Name"]),

            annot=True, linewidths=.5, ax = ax[0],fmt='g', cmap="Reds",

                cbar_kws={'label': 'Count'})

ax[0].set_title('Class Name Count by Division Name - Crosstab\nHeatmap Overall Count Distribution')



sns.heatmap(pd.crosstab(df['Class Name'], df["Division Name"], normalize=True).mul(100).round(0),

            annot=True, linewidths=.5, ax=ax[1],fmt='g', cmap="Reds",

                cbar_kws={'label': 'Percentage %'})

ax[1].set_title('Class Name Count by Division Name - Crosstab\nHeatmap Overall Percentage Distribution')

ax[1].set_ylabel('')

plt.tight_layout(pad=0)

plt.show()



# Heatmaps of Percentage Pivot Table

f, ax = plt.subplots(1,2,figsize=(10, 7), sharey=True)

fsize = 13

sns.heatmap(pd.crosstab(df['Class Name'], df["Division Name"], normalize = 'columns').mul(100).round(0)

            ,annot=True, fmt="g", linewidths=.5, ax=ax[0],cbar=False,cmap="Reds")

ax[0].set_title('Class Name Count by Count - Crosstab\nHeatmap % Distribution by Column', fontsize = fsize)

ax[1] = sns.heatmap(pd.crosstab(df['Class Name'], df["Division Name"], normalize = 'index').mul(100).round(0)

            ,annot=True, fmt="2g", linewidths=.5, ax=ax[1],cmap="Reds",

                cbar_kws={'label': 'Percentage %'})

ax[1].set_title('Class Name Count by Count - Crosstab\nHeatmap % Distribution by Index', fontsize = fsize)

ax[1].set_ylabel('')

plt.tight_layout(pad=0)



# MANUAL NORMALIZE with Applied Lambda on Pandas DataFrame

# ctab = pd.crosstab(df['Class Name'], df["Rating"]).apply(lambda r: r/r.sum(), axis=1).mul(100)
f, axes = plt.subplots(1,4, figsize=(17,4), sharex=False)

xvar = 'Positive Feedback Count'

plotdf = np.log10(df['Positive Feedback Count'])

for i,y in enumerate(["Rating","Department Name","Recommended IND"]):

    for x in set(df[y][df[y].notnull()]):

        sns.kdeplot(plotdf[df[y]==x], label=x, shade=True, ax=axes[i])

    axes[i].set_xlabel("{}\nLog 10".format(xvar))

    axes[i].set_label('Occurrence Density')

    axes[i].set_title('{} Distribution\nby {}'.format(xvar, y))

axes[0].set_ylabel('Occurrence Density')

# Plot 4

sns.kdeplot(plotdf,shade=True,ax=axes[3])

axes[3].set_xlabel("{}\nLog 10".format(xvar))

axes[3].set_title('{} Distribution\n'.format(xvar))

axes[3].legend_.remove()

plt.show()
# Checking inequality difference:

for rec in [0,1]:

    temp = df["Positive Feedback Count"][df["Recommended IND"] == rec]



    print("Recommended is {}".format(rec))

    # 80-20 Rule Confirmation

    print("{}% of Positive Feedback belongs to the top 20% of Reviews with Recommeded = {}".format(

        round(percentage_accumulation(temp, 20)),rec))

    # Gini

    print("Gini Coefficient: {}\n".format(round(gini(temp),2)))
f, axes = plt.subplots(1,3, figsize=(18,4), sharex=False)

for x in set(df["Class Name"][df["Class Name"].notnull()]):

    sns.kdeplot(df['Positive Feedback Count'][df["Class Name"]==x]

                ,label=x, shade=False, ax=axes[0])

    

axes[0].legend_.remove()

axes[0].set_xlabel('{}'.format(xvar))

axes[0].set_title('{} Distribution by {}\n All Data'.format(xvar, "Class Name"))



min_value = 15

for x in set(df["Class Name"][df["Class Name"].notnull()]):

    sns.kdeplot(df['Positive Feedback Count'][(df["Class Name"]==x) &

                                              (df["Positive Feedback Count"] < min_value)]

                ,label=x, shade=False, ax=axes[1])

    

axes[1].legend_.remove()

axes[1].set_xlabel('{}'.format(xvar))

axes[1].set_title('{} Distribution by {}\n Values under {}'.format(xvar, "Class Name", min_value))



for x in set(df["Class Name"][df["Class Name"].notnull()]):

    sns.kdeplot(np.log10(df['Positive Feedback Count']+1)[df["Class Name"]==x]

                ,label=x, shade=False, ax=axes[2])

    

axes[2].legend_.remove()

axes[2].set_xlabel('Log 10 - {}'.format(xvar))

axes[2].set_title('{} Distribution by {}\nAll Data in Log10'.format(xvar, "Class Name"))

plt.show()
f, axes = plt.subplots(1,4, figsize=(16,4), sharex=False)

xvar = "Age"

plotdf = df["Age"]

for i,y in enumerate(["Rating","Department Name","Recommended IND"]):

    for x in set(df[y][df[y].notnull()]):

        sns.kdeplot(plotdf[df[y]==x], label=x, shade=False, ax=axes[i])

    axes[i].set_xlabel("{}".format(xvar))

    axes[i].set_label('Occurrence Density')

    axes[i].set_title('{} Distribution by {}'.format(xvar, y))



for x in set(df["Class Name"][df["Class Name"].notnull()]):

    sns.kdeplot(plotdf[df["Class Name"]==x], label=x, shade=False, ax=axes[3])



axes[3].legend_.remove()

axes[3].set_xlabel('{}'.format(xvar))

axes[0].set_ylabel('Occurrence Density')

axes[3].set_title('{} Distribution by {}'.format(xvar, "Class Name"))

plt.show()
# Normalization is futile here.. But here is a minmax standardization, and a z-score normalization function. 

def minmaxscaler(df):

    return (df-df.min())/(df.max()-df.min())

def zscorenomalize(df):

    return (df - df.mean())/df.std()



g = sns.jointplot(x= df["Positive Feedback Count"], y=df["Age"], kind='reg', color='g')

g.fig.suptitle("Scatter Plot for Age and Positive Feedback Count")

plt.show()
def percentstandardize_barplot(x,y,hue, data, ax=None, order= None):

    """

    Standardize by percentage the data using pandas functions, then plot using Seaborn.

    Function arguments are and extention of Seaborns'.

    """

    sns.barplot(x= x, y=y, hue=hue, ax=ax, order=order,

    data=(data[[x, hue]]

     .reset_index(drop=True)

     .groupby([x])[hue]

     .value_counts(normalize=True)

     .rename('Percentage').mul(100)

     .reset_index()

     .sort_values(hue)))

    plt.title("Percentage Frequency of {} by {}".format(hue,x))

    plt.ylabel("Percentage %")
huevar = "Recommended IND"

f, axes = plt.subplots(1,2,figsize=(12,5))

percentstandardize_barplot(x="Department Name",y="Percentage", hue=huevar,data=df, ax=axes[0])

axes[0].set_title("Percentage Frequency of {}\nby Department Name".format(huevar))

axes[0].set_ylabel("Percentage %")

percentstandardize_barplot(x="Division Name",y="Percentage", hue=huevar,data=df, ax=axes[1])

axes[1].set_title("Percentage Frequency of {}\nby Division Name".format(huevar))

axes[1].set_ylabel("")

plt.show()
xvar = ["Department Name","Division Name"]

huevar = "Rating"

f, axes = plt.subplots(1,2,figsize=(12,5))

percentstandardize_barplot(x=xvar[0],y="Percentage", hue=huevar,data=df, ax=axes[0])

axes[0].set_title("Percentage Frequency of {}\nby {}".format(huevar, xvar[0]))

axes[0].set_ylabel("Percentage %")

percentstandardize_barplot(x=xvar[1],y="Percentage", hue="Rating",data=df, ax=axes[1])

axes[1].set_title("Percentage Frequency of {}\nby {}".format(huevar, xvar[1]))

plt.show()
# Cuttoff Variable

df["Cutoff"] = df["Positive Feedback Count"] >= 40 # Temporary variable for facetgrid

# Facet Grid Plot

g = sns.FacetGrid(df, row = "Cutoff", col="Recommended IND",

                  hue="Rating", size=4, aspect=1.1, sharey=False, sharex=False)

g.map(sns.distplot, "Positive Feedback Count", hist=False)

g.add_legend()

g.axes[0,0].set_ylabel('Density')

g.axes[1,0].set_ylabel('Density')

plt.subplots_adjust(top=0.90)

g.fig.suptitle('Positive Feedback Count by Recommended (Column) and Rating (Color) and Cutoff (Feedback >40 is True)')



# Give cutoff line to each plot.

for x in [0,1]:

    for y in [0,1]:

        g.axes[x,y].axvline(x=40, c="r")



plt.show()

del df["Cutoff"]
huevar = "Rating"

f, axes = plt.subplots(1,2,figsize=(12,5))

sns.countplot(x="Rating", hue="Recommended IND",data=df, ax=axes[0])

axes[0].set_title("Occurrence of {}\nby {}".format(huevar, "Recommended IND"))

axes[0].set_ylabel("Count")

percentstandardize_barplot(x="Rating",y="Percentage", hue="Recommended IND",data=df, ax=axes[1])

axes[1].set_title("Percentage Normalized Occurrence of {}\nby {}".format(huevar, "Recommended IND"))

axes[1].set_ylabel("% Percentage by Rating")

plt.show()
f, axes = plt.subplots(1,3,figsize=(12,5))

rot = 30

df.pivot_table('Rating',

               columns=['Recommended IND']).plot.bar(ax=axes[0],rot=rot)

axes[0].set_title("Average Rating by\nRecommended IND")

df.pivot_table('Rating', index='Division Name',

               columns=['Recommended IND']).plot.bar(ax=axes[1], rot=rot)

axes[1].set_title("Average Rating by Divison Name\nand Recommended IND")

df.pivot_table('Rating', index='Department Name',

               columns=['Recommended IND']).plot.bar(ax=axes[2], rot=rot)

axes[0].set_ylabel("Rating")

axes[2].set_title("Average Rating by Department Name\nand Recommended IND")

f.tight_layout()

plt.show()
temp = (df.groupby('Clothing ID')[["Rating","Recommended IND", "Age"]]

        .aggregate(['count','mean']))

temp.columns = ["Count","Rating Mean","Recommended IND Count",

                "Recommended Mean","Age Count","Age Mean"]

temp.drop(["Recommended IND Count","Age Count"], axis=1, inplace =True)



# Plot Correlation Matrix

f, ax = plt.subplots(figsize=[9,6])

ax = sns.heatmap(temp.corr()

    , annot=True, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'})

ax.set_title("Correlation Matrix for Mean and Count for\nRating,Recommended, and Age\nGrouped by Clothing ID")

plt.show()
g = sns.jointplot(x= "Recommended Mean",y='Rating Mean',data=temp,

                  kind='reg', color='b')

plt.subplots_adjust(top=0.999)

g.fig.suptitle("Rating Mean and Recommended Mean\nGrouped by Clothing ID")

plt.show()
plt.figure(figsize=(7,5))

plt.scatter(temp["Recommended Mean"],temp["Rating Mean"],

            alpha = .8, c =temp["Count"], cmap = 'seismic')

cbar = plt.colorbar() # Color bar. Vive la France!

cbar.set_label('Count', rotation=90)

plt.xlabel("Average Recommended IND")

plt.ylabel("Average Rating")

plt.title("Clothing Piece Frequency (Color) on\nRating and Recommended Mean Scatter")



# Vertical and Horizontal Lines

l = plt.axhline(y=3.3)

l = plt.axvline(x=.55)



# Text

plt.text(.15, 1, "Lower\nQuadrant", ha='left',wrap=True,fontsize=17)

plt.show()



# Descriptives for LOW QUADRANT

temp[(temp["Rating Mean"] < 3.3) | (temp["Recommended Mean"] <= .55)].describe()
key = "Class Name"

temp = (df.groupby(key)[["Rating","Recommended IND", "Age"]]

        .aggregate(['count','mean']))

temp.columns = ["Count","Rating Mean","Recommended Likelihood Count",

                "Recommended Likelihood","Age Count","Age Mean"]

temp.drop(["Recommended Likelihood Count","Age Count"], axis=1, inplace =True)



# Plot Correlation Matrix

f, ax = plt.subplots(figsize=[9,6])

ax = sns.heatmap(temp.corr()

    , annot=True, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'})

ax.set_title("Correlation Coefficient for Mean and Count for\nRating, Recommended Likelihood, and Age\nGrouped by {}".format(key))

plt.show()

print("Class Categories:\n",df["Class Name"].unique())
# Simple Linear Regression Model

model_fit = sm.OLS(temp["Recommended Likelihood"],

               sm.add_constant(temp["Age Mean"])).fit() 

temp['resid'] = model_fit.resid



# Plot

g = sns.jointplot(y="Recommended Likelihood",x='Age Mean',data=temp,

                  kind='reg', color='b')

plt.subplots_adjust(top=0.999)

g.fig.suptitle("Age Mean and Recommended Likelihood\nGrouped by Clothing Class")

plt.ylim(.7, 1.01)



# Annotate Outliers

head = temp.sort_values(by=['resid'], ascending=[False]).head(2)

tail = temp.sort_values(by=['resid'], ascending=[False]).tail(2)



def ann(row):

    ind = row[0]

    r = row[1]

    plt.gca().annotate(ind, xy=( r["Age Mean"], r["Recommended Likelihood"]), 

            xytext=(2,2) , textcoords ="offset points", )



for row in head.iterrows():

    ann(row)

for row in tail.iterrows():

    ann(row)



plt.show()

del head, tail



temp[temp["Recommended Likelihood"] > .95]
pd.set_option('max_colwidth', 500)

df[["Title","Review Text", "Rating"]].sample(7)
from nltk.stem.lancaster import LancasterStemmer

from nltk.stem.porter import PorterStemmer

#ps = LancasterStemmer()

ps = PorterStemmer()



tokenizer = RegexpTokenizer(r'\w+')

stop_words = set(stopwords.words('english'))



def preprocessing(data):

    txt = data.str.lower().str.cat(sep=' ') #1

    words = tokenizer.tokenize(txt) #2

    words = [w for w in words if not w in stop_words] #3

    #words = [ps.stem(w) for w in words] #4

    return words
# Pre-Processing

SIA = SentimentIntensityAnalyzer()

df["Review Text"]= df["Review Text"].astype(str)



# Applying Model, Variable Creation

df['Polarity Score']=df["Review Text"].apply(lambda x:SIA.polarity_scores(x)['compound'])

df['Neutral Score']=df["Review Text"].apply(lambda x:SIA.polarity_scores(x)['neu'])

df['Negative Score']=df["Review Text"].apply(lambda x:SIA.polarity_scores(x)['neg'])

df['Positive Score']=df["Review Text"].apply(lambda x:SIA.polarity_scores(x)['pos'])



# Converting 0 to 1 Decimal Score to a Categorical Variable

df['Sentiment']=''

df.loc[df['Polarity Score']>0,'Sentiment']='Positive'

df.loc[df['Polarity Score']==0,'Sentiment']='Neutral'

df.loc[df['Polarity Score']<0,'Sentiment']='Negative'
huevar = "Recommended IND"

xvar = "Sentiment"

f, axes = plt.subplots(1,2,figsize=(12,5))

sns.countplot(x=xvar, hue=huevar,data=df, ax=axes[0], order=["Negative","Neutral","Positive"])

axes[0].set_title("Occurence of {}\nby {}".format(xvar, huevar))

axes[0].set_ylabel("Count")

percentstandardize_barplot(x=xvar,y="Percentage", hue=huevar,data=df, ax=axes[1])

axes[1].set_title("Percentage Normalized Occurence of {}\nby {}".format(xvar, huevar))

axes[1].set_ylabel("% Percentage by {}".format(huevar))

plt.show()
f, axes = plt.subplots(2,2, figsize=[9,9])

sns.countplot(x="Sentiment", data=df, ax=axes[0,0], order=["Negative","Neutral","Positive"])

axes[0,0].set_xlabel("Sentiment")

axes[0,0].set_ylabel("Count")

axes[0,0].set_title("Overall Sentiment Occurrence")



sns.countplot(x="Rating", data=df, ax=axes[0,1])

axes[0,1].set_xlabel("Rating")

axes[0,1].set_ylabel("")

axes[0,1].set_title("Overall Raiting Occurrence")



percentstandardize_barplot(x="Rating",y="Percentage",hue="Sentiment",data=df, ax=axes[1,0])

axes[1,0].set_xlabel("Rating")

axes[1,0].set_ylabel("Percentage %")

axes[1,0].set_title("Standardized Percentage Raiting Frequency\nby Sentiment")



percentstandardize_barplot(x="Sentiment",y="Percentage",hue="Rating",data=df, ax=axes[1,1])

axes[1,1].set_ylabel("Occurrence Frequency")

axes[1,1].set_title("Standardized Percentage Sentiment Frequency\nby Raiting")

axes[1,1].set_xlabel("Sentiment")

axes[1,1].set_ylabel("")



f.suptitle("Distribution of Sentiment Score and Rating for Customer Reviews", fontsize=14)

f.tight_layout()

f.subplots_adjust(top=0.92)

plt.show()
# Tweakable Variables (Note to Change Order Arguement if Xvar is changed)

xvar = "Sentiment"

huevar = "Department Name"

rowvar = "Recommended IND"



# Plot

f, axes = plt.subplots(2,2,figsize=(10,10), sharex=False,sharey=False)

for i,x in enumerate(set(df[rowvar][df[rowvar].notnull()])):

    percentstandardize_barplot(x=xvar,y="Percentage", hue=huevar,data=df[df[rowvar] == x],

                 ax=axes[i,0], order=["Negative","Neutral","Positive"])

    percentstandardize_barplot(x=xvar,y="Percentage", hue="Rating",data=df[df[rowvar] == x],

                 ax=axes[i,1], order=["Negative","Neutral","Positive"])



# Plot Aesthetics

axes[1,0].legend_.remove()

axes[1,1].legend_.remove()

axes[0,1].set_ylabel("")

axes[1,1].set_ylabel("")

axes[0,0].set_xlabel("")

axes[0,1].set_xlabel("")

axes[0,0].set_ylabel("Recommended = FALSE\nPercentage %")

axes[1,0].set_ylabel("Recommended = TRUE\nPercentage %")

axes[1,1].set_title("")



# Common title and ylabel

f.text(0.0, 0.5, 'Subplot Rows\nSliced by Recommended', va='center', rotation='vertical', fontsize=12)

f.suptitle("Review Sentiment by Department Name and Raiting\nSubplot Rows Slice Data by Recommended", fontsize=16)

f.tight_layout()

f.subplots_adjust(top=0.93)

plt.show()
# Plot Correlation Matrix

f, ax = plt.subplots(figsize=[9,6])

ax = sns.heatmap(df.corr(), annot=True,

                 fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'})

ax.set_title("Correlation Matrix for All Variables")

plt.show()



# Sentiment Positivity Score by Positive Feedback Count

ax = sns.jointplot(x= df["Positive Feedback Count"], y=df["Positive Score"], kind='reg', color='r')

plt.show()
stopwords = set(STOPWORDS)

size = (10,7)



def cloud(text, title, stopwords=stopwords, size=size):

    """

    Function to plot WordCloud

    Includes: 

    """

    # Setting figure parameters

    mpl.rcParams['figure.figsize']=(10.0,10.0)

    mpl.rcParams['font.size']=12

    mpl.rcParams['savefig.dpi']=100

    mpl.rcParams['figure.subplot.bottom']=.1 

    

    # Processing Text

    # Redundant when combined with my Preprocessing function

    wordcloud = WordCloud(width=1600, height=800,

                          background_color='black',

                          stopwords=stopwords,

                         ).generate(str(text))

    

    # Output Visualization

    fig = plt.figure(figsize=size, dpi=80, facecolor='k',edgecolor='k')

    plt.imshow(wordcloud,interpolation='bilinear')

    plt.axis('off')

    plt.title(title, fontsize=50,color='y')

    plt.tight_layout(pad=0)

    plt.show()

    

# Frequency Calculation [One-Gram]

def wordfreqviz(text, x):

    word_dist = nltk.FreqDist(text)

    top_N = x

    rslt = pd.DataFrame(word_dist.most_common(top_N),

                    columns=['Word', 'Frequency']).set_index('Word')

    matplotlib.style.use('ggplot')

    rslt.plot.bar(rot=0)



def wordfreq(text, x):

    word_dist = nltk.FreqDist(text)

    top_N = x

    rslt = pd.DataFrame(word_dist.most_common(top_N),

                    columns=['Word', 'Frequency']).set_index('Word')

    return rslt
# Modify Stopwords to Exclude Class types, suchs as "dress"

new_stop = set(STOPWORDS)

new_stop.update([x.lower() for x in list(df["Class Name"][df["Class Name"].notnull()].unique())]

                + ["dress", "petite"])



# Cloud

cloud(text= df.Title[df.Title.notnull()].astype(str).values,

      title="Titles",

      stopwords= new_stop,

      size = (7,4))
# Highly Raited

title ="Highly Rated Comments"

temp = df['Review Text'][df.Rating.astype(int) >= 3]



# Modify Stopwords to Exclude Class types, suchs as "dress"

new_stop = set(STOPWORDS)

new_stop.update([x.lower() for x in list(df["Class Name"][df["Class Name"].notnull()].unique())]

                + ["dress", "petite"])



# Cloud

cloud(text= temp.values, title=title,stopwords= new_stop)



# Bar Chart

wordfreq(preprocessing(temp),20).plot.bar(rot=45, legend=False,figsize=(15,5), color='g',

                          title= title)

plt.ylabel("Occurrence Count")

plt.xlabel("Most Frequent Words")

plt.show()



# Low Raited

title ="Most Frequent Words in Low Rated Comments"

temp = df['Review Text'][df.Rating.astype(int) < 3]



# Modify Stopwords to Exclude Class types, suchs as "dress"

new_stop = set(STOPWORDS)

new_stop.update([x.lower() for x in list(df["Class Name"][df["Class Name"].notnull()].unique())]

                + ["dress", "petite", "skirt","shirt"])



# Cloud

cloud(temp.values, title= title, stopwords = new_stop)
department_set = df["Department Name"][df["Department Name"].notnull()].unique()

division_set = df["Division Name"][df["Division Name"].notnull()].unique()

def cloud_by_category(data, category, subclass):

    """

    Function to create a wordcloud by class and subclass

    Category signifies the column variable

    Subclass refers to the specific value within the categorical variable

    """

    new_stop = set(STOPWORDS)

    new_stop.update([x.lower() for x in list(data["Class Name"][data["Class Name"].notnull()].unique())]

                   + [x.lower() for x in list(data["Department Name"][data["Department Name"].notnull()].unique())]

                   + ["dress", "petite", "jacket","top"])



    # Cloud

    cloud(text= data["Review Text"][data[category]== subclass],

          title="{}".format(subclass),

          stopwords= new_stop,

          size = (10,6))

    

# Plot

cloud_by_category(df, "Division Name", division_set[0])

cloud_by_category(df, "Division Name", division_set[1])

cloud_by_category(df, "Division Name", division_set[2])
## Helper Functions

from nltk.util import ngrams

from collections import Counter

def get_ngrams(text, n):

    n_grams = ngrams((text), n)

    return [ ' '.join(grams) for grams in n_grams]



def gramfreq(text,n,num):

    # Extracting bigrams

    result = get_ngrams(text,n)

    # Counting bigrams

    result_count = Counter(result)

    # Converting to the result to a data frame

    df = pd.DataFrame.from_dict(result_count, orient='index')

    df = df.rename(columns={'index':'words', 0:'frequency'}) # Renaming index column name

    return df.sort_values(["frequency"],ascending=[0])[:num]



def gram_table(data, gram, length):

    out = pd.DataFrame(index=None)

    for i in gram:

        table = pd.DataFrame(gramfreq(preprocessing(data),i,length).reset_index())

        table.columns = ["{}-Gram".format(i),"Occurrence"]

        out = pd.concat([out, table], axis=1)

    return out
print("Non-Recommended Items")

gram_table(data= df['Review Text'][df["Recommended IND"].astype(int) == 0], gram=[1,2,3,4,5], length=30)
print("Recommended Items")

gram_table(data= df['Review Text'][df["Recommended IND"].astype(int) == 1], gram=[1,2,3,4,5], length=30)
df['tokenized'] = df["Review Text"].astype(str).str.lower() # Turn into lower case text

df['tokenized'] = df.apply(lambda row: tokenizer.tokenize(row['tokenized']), axis=1) # Apply tokenize to each row

df['tokenized'] = df['tokenized'].apply(lambda x: [w for w in x if not w in stop_words]) # Remove stopwords from each row

df['tokenized'] = df['tokenized'].apply(lambda x: [ps.stem(w) for w in x]) # Apply stemming to each row

all_words = nltk.FreqDist(preprocessing(df['Review Text'])) # Calculate word occurrence from whole block of text



vocab_count = 200

word_features= list(all_words.keys())[:vocab_count] # 2000 most recurring unique words

print("Number of words columns (One Hot Encoding): {}".format(len(all_words)))
# Tuple

labtext= list(zip(df.tokenized, (df["Recommended IND"]))) 



# Function to create model features

# for each review, records which unique words out of the whole text body are present

def find_features(document):

    words = set(document)

    features = {}

    for w in word_features:

        features[w] = (w in words)



    return features

# Apply function to data

featuresets = [(find_features(text), LABEL) for (text, LABEL) in labtext]

len(featuresets)



# Train/Test

training_set = featuresets[:15000]

testing_set = featuresets[15000:]
# Posterior = prior_occurrence * likelihood / evidence

classifier = nltk.NaiveBayesClassifier.train(training_set)



# Output

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

print(classifier.show_most_informative_features(40))
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.pipeline import FeatureUnion

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split

from sklearn import metrics

import scikitplot as skplt

import eli5
vect = TfidfVectorizer()

vect.fit(df["Review Text"])

X = vect.transform(df["Review Text"])
y = df["Recommended IND"].copy()



X_train, X_valid, y_train, y_valid = train_test_split(

    X, y, test_size=0.20, random_state=23, stratify=y)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

print("Train Set Accuracy: {}".format(metrics.accuracy_score(model.predict(X_train), y_train)))

print("Train Set ROC: {}\n".format(metrics.roc_auc_score(model.predict(X_train), y_train)))



print("Validation Set Accuracy: {}".format(metrics.accuracy_score(model.predict(X_valid), y_valid)))

print("Validation Set ROC: {}".format(metrics.roc_auc_score(model.predict(X_valid), y_valid)))
print(metrics.classification_report(model.predict(X_valid), y_valid))
# Confusion Matrix

skplt.metrics.plot_confusion_matrix(model.predict(X_valid), y_valid, normalize=True)

plt.show()
target_names = ["Not Recommended","Recommended"]

eli5.show_weights(model, vec=vect, top=100,

                  target_names=target_names)
for iteration in range(15):

    samp = random.randint(1,df.shape[0])

    print("Real Label: {}".format(df["Recommended IND"].iloc[samp]))

    display(eli5.show_prediction(model,df["Review Text"].iloc[samp], vec=vect,

                         target_names=target_names))
import lightgbm as lgb



print("Light Gradient Boosting Classifier: ")

lgbm_params = {

        "objective": "binary",

        'metric': {'auc'},

        "boosting_type": "gbdt",

        "num_threads": 4,

        "bagging_fraction": 0.8,

        "feature_fraction": 0.8,

        "learning_rate": 0.1,

        "num_leaves": 31,

        "min_split_gain": .1,

        "reg_alpha": .1

    }



modelstart= time.time()

# LGBM Dataset Formatting 

lgtrain = lgb.Dataset(X_train, y_train,

                feature_name=vect.get_feature_names())

lgvalid = lgb.Dataset(X_valid, y_valid,

                feature_name=vect.get_feature_names())
# Go Go Go

lgb_clf = lgb.train(

    lgbm_params,

    lgtrain,

    num_boost_round=2000,

    valid_sets=[lgtrain, lgvalid],

    valid_names=['train','valid'],

    early_stopping_rounds=150,

    verbose_eval=100

)
valid_pred = lgb_clf.predict(X_valid)

_thresh = []

for thresh in np.arange(0.1, 0.501, 0.01):

    _thresh.append([thresh, metrics.f1_score(y_valid, (valid_pred>thresh).astype(int))])

#     print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_valid, (valid_pred>thresh).astype(int))))



_thresh = np.array(_thresh)

best_id = _thresh[:,1].argmax()

best_thresh = _thresh[best_id][0]

print("Best Threshold: {}\nF1 Score: {}".format(best_thresh, _thresh[best_id][1]))
import shap

shap.initjs()



non_sparse = pd.DataFrame(vect.transform(df['Review Text']).toarray(), columns = vect.get_feature_names())

print(non_sparse.shape)



explainer = shap.TreeExplainer(lgb_clf)

shap_values = explainer.shap_values(non_sparse)
# summarize the effects of all the features

shap.summary_plot(shap_values, non_sparse)
# visualize the first prediction's explanation

for iteration in range(15):

    samp = random.randint(1,df.shape[0])

    print("Real Label: {}".format(df["Recommended IND"].iloc[samp]))

    display(df["Review Text"].iloc[samp])

    display(shap.force_plot(explainer.expected_value, shap_values[samp,:], non_sparse.iloc[samp,:]))