import numpy as np 

import pandas as pd

import os

import matplotlib.pyplot as plt

from pylab import rcParams

from cycler import cycler

from scipy import mean

from collections import Counter

# print(os.listdir("../input"))

authors = pd.read_csv("../input/Authors.csv")

papers_authors = pd.read_csv("../input/PaperAuthors.csv")

papers = pd.read_csv("../input/Papers.csv")

# MatPlotLib PyPlot parameters 

rcParams['figure.figsize'] = (14, 6)

rcParams['axes.titlesize'] = 'xx-large'

rcParams['axes.labelsize'] = 'x-large'

# def get_hex(rgb): 

#     return '#%02x%02x%02x' % (int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255))

# colors = [get_hex(plt.get_cmap("Pastel1")(i)) for i in range(10)]

colors = ['#1BE7FF', '#6EEB83', '#E4FF1A', '#E8AA14', '#FF5714']

rcParams['axes.prop_cycle'] = cycler('color', colors)
plt.figure(figsize=(15,4))

ax = papers['EventType'].value_counts(sort=True).plot.barh(width=0.666, edgecolor='lightgrey')

plt.title("Number of papers in each category")

plt.tick_params(left=False)

for cat, val in enumerate(papers['EventType'].value_counts(sort=True)):

    ax.text(3+val, cat, str(val), fontsize='x-large', horizontalalignment='left', verticalalignment='center')

for tick in ax.yaxis.get_major_ticks():

    tick.label.set_fontsize('x-large') 

for spine in ax.spines: 

    ax.spines[spine].set_visible(False)

plt.xticks([]) # plt.xlabel("Number of papers")

plt.show()
authors['Publications'] = 0

papers['Authors'] = 0

for i, paper_author in papers_authors.iterrows(): 

    authors.loc[authors['Id'] == paper_author['AuthorId'], 'Publications'] += 1

    papers.loc[papers['Id'] == paper_author['PaperId'], 'Authors'] += 1

authors.head(3)
author_dict = {aut['Name']: aut['Publications'] for i, aut in authors.iterrows()}

paper_dict = {pub['Title']: pub['Authors'] for i, pub in papers.iterrows()}

publication_distribution = sorted(authors['Publications'], reverse=True)

print(f"{len(publication_distribution)} publication-author links were found (multiple authors can exist for a single publication, total number of publications: {len(papers)})")

most_prolific = sorted(author_dict, key=lambda x: author_dict[x], reverse=True)[0]

print(f"The author with his name on the most publications was {most_prolific} with {author_dict[most_prolific]} articles published.")

lone_publications = len([val for val in author_dict.values() if val == 1])

not_lone_publications = len([val for val in author_dict.values() if val != 1])

print(f"{lone_publications} authors only published one document each ({not_lone_publications} authors published more than one), the average author published {mean(authors['Publications']):.2f} papers")

most_authors = sorted(paper_dict, key=lambda x: paper_dict[x], reverse=True)[0]

print(f"The paper with the most authors was '{most_authors}' and had {paper_dict[most_authors]} authors.")

solo_papers = len([val for val in paper_dict.values() if val == 1])

team_papers = len([val for val in paper_dict.values() if val != 1])

print(f"{solo_papers} papers were published by a solo author while {team_papers} papers were published by multiple authors, the average paper was authored by {mean(papers['Authors']):.2f} researchers")
plt.figure(figsize=[23,7])

n_bins = 15

x = np.arange(len(publication_distribution))

ax = plt.bar(x, publication_distribution)

plt.ylim(top=max(publication_distribution))

plt.xlim(0, max(x))

plt.xticks([])

plt.ylabel("Number of publications")

plt.xlabel("Publisher")

plt.title('Distribution of publications per publisher')

ax = plt.gca()

for spine in ax.spines: 

    ax.spines[spine].set_visible(False)

plt.show()
plt.figure(figsize=[20,7])

author_dict = dict(Counter(publication_distribution))

num_authors = [author_dict[papers_written] for papers_written in range(1, 8)]

C = author_dict[1] #number of single-paper publications 

n = 2 #field specific, for research publications, Lotka's Law specifies n=2

x_values = np.arange(0.9, 7.1, 0.01)

y_values = list(map(lambda x: C/(x**n), x_values)) # Lotka's Law

plt.plot(range(1, 8), num_authors, 'D', markersize=9, markeredgecolor='grey', label="Actual")

plt.plot(x_values, y_values, lw=2, zorder=0, label="Implied (Lotka's Law)")

plt.ylim(bottom=0, top=max(num_authors)*1.1)

plt.xlim(min(x_values), max(x_values))

plt.ylabel("Number of authors")

plt.xlabel("Papers written")

plt.title("Distribution of author production v.s. Lotka's Law")

plt.legend(loc='upper right')

plt.show()
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

papers['Title_Stemmed'] = papers['Title'].map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))

papers.head()