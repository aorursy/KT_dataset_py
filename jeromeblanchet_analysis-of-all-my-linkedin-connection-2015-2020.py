# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Assessment of my LinkedIn Connections (2015-2020) - by Jerome Blanchet

#Thank you to Guillaume Chevalier for his amazing github repository

#Source Code below from Guillaume Chevalier, with few customization and data cleansing of mine

#https://github.com/guillaume-chevalier/LinkedIn-Connections-Growth-Analysis/blob/master/plot.py
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from statsmodels.tsa.seasonal import seasonal_decompose
FIGSIZE = (16, 9)

FONT = {"family": "Share Tech Mono", "weight": "normal", "size": 16}

tds = "#0073b1"

week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
df = pd.read_csv('../input/linkedin-connection-list/Connections.csv')
df['Connected On'] = pd.to_datetime(df['Connected On'])

df.set_index('Connected On', inplace=True, drop=True)

df.sort_index(inplace=True)

df = df.assign(added=1)
#because recent connections are always more significant.

df.tail(60)
#Source Code from Guillaume Chevalier

#https://github.com/guillaume-chevalier/LinkedIn-Connections-Growth-Analysis/blob/master/plot.py



def plot_weekly_connection(df):

    weekly = df[["added"]].resample("W").sum()

    #style = dict(size=10, color='gray')



    fig, ax = plt.subplots(figsize=FIGSIZE)



    plt.plot(weekly.index, weekly.added, c=tds)



    plt.title("Raw LinkedIn connections evolution\n for Jerome Blanchet (2015-2020)\nNo Seasonal Adjustment", 

    fontdict=FONT, y=1.2)

    

    plt.ylabel("Nb weekly connections", fontdict=FONT)

    

    #ax.text('2012-01-01', 500, "New Year's Day", **style)

    

    ax.set_frame_on(False)

    plt.grid(True)

    plt.show()





def plot_cumsum(df):

    cumsum = df.added.cumsum()



    fig, ax = plt.subplots(figsize=FIGSIZE)



    plt.plot(cumsum.index, cumsum.values, c=tds)



    plt.title("LinkedIn connections evolution (cumulated)\n for Jerome Blanchet", fontdict=FONT, y=1.2)

    plt.ylabel("Nb connections", fontdict=FONT)



    ax.set_frame_on(False)

    plt.grid(True)

    plt.show()





def violins_prep(tmp):

    tmp = tmp.resample("D").sum()

    tmp = tmp.assign(dow=tmp.index.dayofweek.values).sort_values("dow")

    return tmp.assign(dow_str=tmp.dow.apply(lambda d: week[d]))





def plot_violins(df):

    violins = violins_prep(df[["added"]])



    fig, ax = plt.subplots(figsize=(20, 8))

    ax = sns.violinplot(x="dow_str", y="added", data=violins, color=tds, cut=0, ax=ax)



    plt.title("LinkedIn connections distribution per day of week\n for Jerome Blanchet", fontdict=FONT, y=1.2)

    plt.xlabel("Week day", fontdict=FONT)

    plt.ylabel("Nb daily connections", fontdict=FONT)



    ax.set_frame_on(False)

    plt.grid(True)

    plt.show()





def plot_bar_column(df, col):

    fnames = df[col].value_counts().head(30)

    plot_fnames(fnames,col)





def plot_nlp_cv(df):

    tfidf = CountVectorizer(ngram_range=(1, 3), stop_words='english')

    cleaned_positions = list(df["Position"].fillna(""))

    res = tfidf.fit_transform(cleaned_positions)

    res = res.toarray().sum(axis=0)



    fnames = pd.DataFrame(

        list(sorted(zip(res, tfidf.get_feature_names())))[-30:],

        columns=["Position by Words Freq", "Words"]

    )[::-1] 

    plot_fnames(fnames, "Position by Words Freq", "Words")





def plot_fnames(fnames, col, index="index"):

    fnames = fnames.reset_index()



    fig, ax = plt.subplots(figsize=FIGSIZE)



    plt.bar(

        x=fnames.index,

        height=fnames[col],

        color=tds,

        alpha=0.5

    )



    plt.title("{} distribution".format(col), fontdict=FONT, y=1.2)

    plt.xticks(

        fnames.index,

        fnames[index].str.capitalize(),

        rotation=65,

        ha="right",

        size=FONT["size"],

    )



    plt.ylabel("Nb occurences", fontdict=FONT)

    plt.yticks()#[0, 5, 10, 15, 20])

    ax.set_frame_on(False)

    plt.grid(True)



    plt.show()
#Custom data cleaning for repetitive company names
#The company I used to work for has several long title on LinkedIn

#Shorter and consistent name is better

#Apple is simply Apple right!

fnames = df['Company'].value_counts().head(50)

print(fnames)
#Let's build the dictionary and map everything



df.Company.replace({'CMHC - SCHL':'CMHC',

'Canada Mortgage and Housing Corporation':'CMHC',

'Canada Mortgage and Housing Corporation (CMHC)':'CMHC',

'Canada Mortgage and Housing Corporation CMHC / Société canadienne d\'hypothèques et de logement SCHL':'CMHC',

'Société canadienne d\'hypothèques et de logement(SCHL) Canada Mortgage and Housing Corporation (CMHC)':'CMHC',

'Canada Mortgage and Housing Corporation (CMHC) Société canadienne d\'hypothèques et de logement(SCHL)':'CMHC',

'Export Development Canada | Exportation et développement Canada - EDC':'EDC',

'Immigration, Refugees and Citizenship Canada / Immigration, Réfugiés et Citoyenneté Canada':'Immigration Canada',

'Transport Canada - Transports Canada':'Transport Canada'

}, inplace=True)
#Way better now and easier to read!!!!!!

fnames = df['Company'].value_counts().head(50)

print(fnames)
plot_weekly_connection(df)
plot_cumsum(df)
plot_violins(df)
plot_bar_column(df, "First Name")
plot_bar_column(df, "Company")
#Company distribution without CMHC



def plot_bar_columnn(df, col):

    fnames = df[col].value_counts().head(39)[1:]

    plot_fnames(fnames,col)



plot_bar_columnn(df, "Company")
plot_bar_column(df, "Position")
plot_nlp_cv(df)