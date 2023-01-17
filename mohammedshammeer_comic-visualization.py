import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing the datasets

dc = pd.read_csv("../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv")

marvel = pd.read_csv("../input/fivethirtyeight-comic-characters-dataset/marvel-wikia-data.csv")
dc.head()
marvel.head()
# Bar Plot

plt.figure(figsize=(20, 4))

sns.set_context('paper', font_scale = .5, rc={"grid.linewidth": 0.5})

sns.set_style("whitegrid")

sns.barplot(x=dc["YEAR"], y=dc["APPEARANCES"])

plt.title("DC Appearances based on Year", fontsize=25)

plt.xlabel("Year", fontsize=20)

plt.ylabel("Appearances", fontsize=20)

plt.yticks(fontsize=8);
# Bar Plot

plt.figure(figsize=(20, 4))

sns.set_context('paper', font_scale = .5, rc={"grid.linewidth": 0.5})

sns.set_style("whitegrid")

sns.barplot(x=marvel["Year"], y=marvel["APPEARANCES"])

plt.title("Marvel Appearances based on Year", fontsize=25)

plt.xlabel("Year", fontsize=20)

plt.ylabel("Appearances", fontsize=20)

plt.yticks(fontsize=8);
f = plt.figure(figsize=(16, 25))

gs = f.add_gridspec(1, 2)

with sns.axes_style("white"):

    sns.set()

    sns.set_context('paper', font_scale = .5, rc={"grid.linewidth": 0.5})

    sns.set_style("whitegrid")

    g = sns.catplot(x="YEAR", y="APPEARANCES", hue="SEX",

                palette="hls", linestyles=["-", "--", "-.", ":"],

                kind="point", data=dc, height=6, aspect=20/6, margin_titles=True, ci=None)

    (g.despine(left=True))  

    plt.title("DC Data based on Sex", fontsize=25)

    plt.xlabel("Year", fontsize=20)

    plt.ylabel("Appearances", fontsize=20)

    plt.tight_layout()    

with sns.axes_style("white"):

    sns.set()

    sns.set_context('paper', font_scale = .5, rc={"grid.linewidth": 0.5})

    sns.set_style("whitegrid")

    g = sns.catplot(x="Year", y="APPEARANCES", hue="SEX",

                palette="hls", linestyles=["-", "--", "-.", ":"],

                kind="point", data=marvel, height=6, aspect=20/6, margin_titles=True, ci=None)

    (g.despine(left=True))  

    plt.title("Marvel Data based on Sex", fontsize=25)

    plt.xlabel("Year", fontsize=20)

    plt.ylabel("Appearances", fontsize=20)

    plt.tight_layout()
gs = f.add_gridspec(1, 2)

with sns.axes_style("white"):

    sns.set()

    sns.set_context('paper', font_scale = .5, rc={"grid.linewidth": 0.5})

    sns.set_style("whitegrid")

    g = sns.catplot(x="YEAR", y="APPEARANCES", hue="ALIGN",

                palette="hls", linestyles=["-", "--", "-.", ":"],

                kind="point", data=dc, height=6, aspect=20/6, margin_titles=True, ci=None)

    (g.despine(left=True))  

    plt.title("DC Data based on their Nature", fontsize=25)

    plt.xlabel("Year", fontsize=20)

    plt.ylabel("Appearances", fontsize=20)

    plt.tight_layout()

with sns.axes_style("white"):

    sns.set()

    sns.set_context('paper', font_scale = .5, rc={"grid.linewidth": 0.5})

    sns.set_style("whitegrid")

    g = sns.catplot(x="Year", y="APPEARANCES", hue="ALIGN",

                palette="hls", linestyles=["-", "--", "-.", ":"],

                kind="point", data=marvel, height=6, aspect=20/6, margin_titles=True, ci=None)

    (g.despine(left=True))  

    plt.title("Marvel Data based on their Nature", fontsize=25)

    plt.xlabel("Year", fontsize=20)

    plt.ylabel("Appearances", fontsize=20)

    plt.tight_layout()
# Scatter Plot for better understanding

sorted_dc = dc.sort_values('APPEARANCES', na_position='last')

sorted_dc = sorted_dc.iloc[:6400, :]

sorted_marvel = marvel.sort_values('APPEARANCES', na_position='last')

sorted_marvel = sorted_marvel.iloc[:15000, :]

f = plt.figure(figsize=(16, 25))

gs = f.add_gridspec(1, 2)

with sns.axes_style("white"):

    ax = f.add_subplot(gs[0, 0])

    sns.scatterplot(x=sorted_dc['YEAR'], y=sorted_dc['APPEARANCES'], hue=sorted_dc['SEX'])

    plt.title("DC Data based on Sex", fontsize=25)

    plt.xlabel('Year', fontsize=20)

with sns.axes_style("white"):

    ax = f.add_subplot(gs[0, 1])

    sns.scatterplot(x=sorted_marvel['Year'], y=sorted_marvel['APPEARANCES'], hue=sorted_marvel['SEX'])

    plt.title("Marvel Data based on Sex", fontsize=25)

    plt.xlabel('Year', fontsize=20)
# Scatter Plot for better understanding

f = plt.figure(figsize=(16, 25))

gs = f.add_gridspec(1, 2)

with sns.axes_style("white"):

    ax = f.add_subplot(gs[0, 0])

    sns.scatterplot(x=sorted_dc['YEAR'], y=sorted_dc['APPEARANCES'], hue=sorted_dc['ALIGN'])

    plt.title("DC Data based on Characters", fontsize=25)

    plt.xlabel('Year', fontsize=20)

with sns.axes_style("white"):

    ax = f.add_subplot(gs[0, 1])

    sns.scatterplot(x=sorted_marvel['Year'], y=sorted_marvel['APPEARANCES'], hue=sorted_marvel['ALIGN'])

    plt.title("Marvel Data based on Characters", fontsize=25)

    plt.xlabel('Year', fontsize=20)
gs = f.add_gridspec(1, 2)

with sns.axes_style("white"):

    sns.set()

    sns.set_context('paper', font_scale = .5, rc={"grid.linewidth": 0.5})

    sns.set_style("whitegrid")

    g = sns.catplot(x="YEAR", y="APPEARANCES", hue="EYE",

                palette="hls",

                kind="point", data=dc, height=6, aspect=20/6, margin_titles=True, ci=None) 

    plt.title("DC Data based on Eye Color", fontsize=25)

    plt.xlabel("Year", fontsize=20)

    plt.ylabel("Appearances", fontsize=20)

    plt.tight_layout()

with sns.axes_style("white"):

    sns.set()

    sns.set_context('paper', font_scale = .5, rc={"grid.linewidth": 0.5})

    sns.set_style("whitegrid")

    g = sns.catplot(x="Year", y="APPEARANCES", hue="EYE",

                palette="hls",

                kind="point", data=marvel, height=6, aspect=20/6, margin_titles=True, ci=None)

    plt.title("Marvel Data based on Eye Color", fontsize=25)

    plt.xlabel("Year", fontsize=20)

    plt.ylabel("Appearances", fontsize=20)

    plt.tight_layout()
gs = f.add_gridspec(1, 2)

with sns.axes_style("white"):

    sns.set()

    sns.set_context('paper', font_scale = .5, rc={"grid.linewidth": 0.5})

    sns.set_style("whitegrid")

    g = sns.catplot(x="Year", y="APPEARANCES", hue="HAIR",

                palette="hls",

                kind="point", data=marvel, height=6, aspect=20/6, margin_titles=True, ci=None) 

    plt.title("Marvel Data based on Hair Color", fontsize=25)

    plt.xlabel("Year", fontsize=20)

    plt.ylabel("Appearances", fontsize=20)

    plt.tight_layout()

with sns.axes_style("white"):

    sns.set()

    sns.set_context('paper', font_scale = .5, rc={"grid.linewidth": 0.5})

    sns.set_style("whitegrid")

    g = sns.catplot(x="YEAR", y="APPEARANCES", hue="HAIR",

                palette="hls",

                kind="point", data=dc, height=6, aspect=20/6, margin_titles=True, ci=None)

    plt.title("DC Data based on Hair Color", fontsize=25)

    plt.xlabel("Year", fontsize=20)

    plt.ylabel("Appearances", fontsize=20)

    plt.tight_layout()