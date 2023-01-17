# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import seaborn as sns

import random

from os import path

import nltk



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
debate = pd.read_csv('../input/debate.csv',encoding = 'iso-8859-1')

debate.head(10)
debate = debate.loc[debate.Date == "2016-09-26"]

debate.Speaker.drop_duplicates()
CLINTON = "Clinton"

TRUMP = "Trump"
trumpLines = debate[debate['Speaker'].str.contains(TRUMP)]
from nltk.corpus import stopwords

from nltk import word_tokenize

stopword_set = set(stopwords.words('english'))
len(stopwords.words('english'))
# Tokenize our lines

trumpLines_tokenized = [word_tokenize(line.lower()) for line in trumpLines['Text']]
#borrowed from Research gate

import string

def clear_punctuation(s):

    clear_string = ""

    for symbol in s:

        if symbol not in string.punctuation:

            clear_string += symbol

    return clear_string

trumpLines_tokenized = [word_tokenize(clear_punctuation(line.lower())) for line in trumpLines['Text']]
trump_no_stops = [[word for word in line if word not in stopword_set] for line in trumpLines_tokenized]
len(trump_no_stops[0])/len(trumpLines_tokenized[0])
from collections import Counter

trump_no_stops_flat= [item for linelist in trump_no_stops for item in linelist]
tdf_trump_counts = Counter(trump_no_stops_flat)
tdf_trump_t25= tdf_trump_counts.most_common(25)
plotListTrumpLabel = [label for label, count in tdf_trump_t25]

plotListTrumpCount = [count for label, count in tdf_trump_t25]
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.ticker import *

sns.set(style="white", context="talk")





# Set up the matplotlib figure

f, (ax1) = plt.subplots(1, 1, figsize=(8, 6), sharex=True)



# Use our extracted data

x = plotListTrumpLabel

y1 = plotListTrumpCount

sns.barplot(x, y1, palette="Set3", ax=ax1)

ax1.set_ylabel("Number of utterances")

ax1.set_xticklabels(x, rotation =90)



# Finalize the plot

sns.despine(bottom=True)

plt.setp(f.axes, yticks=[])

plt.tight_layout(h_pad=3)
plotListTrumpLabel
hillaryLines = debate[debate['Speaker'].str.contains(CLINTON)]

hillaryLines_tokenized = [word_tokenize(clear_punctuation(line.lower())) for line in hillaryLines['Text']]

hillary_no_stops = [[word for word in line if word not in stopword_set] for line in hillaryLines_tokenized]

hillary_no_stops_flat= [item for linelist in hillary_no_stops for item in linelist]

tdf_hillary_counts = Counter(hillary_no_stops_flat)
tdf_hillary_t25= tdf_hillary_counts.most_common(25)

plotListHillaryLabel = [label for label, count in tdf_hillary_t25]

plotListHillaryCount = [count for label, count in tdf_hillary_t25]
sns.set(style="white", context="talk")





# Set up the matplotlib figure

f, (ax1) = plt.subplots(1, 1, figsize=(8, 6), sharex=True)



# Use our extracted data

x = plotListHillaryLabel

y1 = plotListHillaryCount

sns.barplot(x, y1, palette="Set3", ax=ax1)

ax1.set_ylabel("Number of utterances")

ax1.set_xticklabels(x, rotation =90)



# Finalize the plot

sns.despine(bottom=True)

plt.setp(f.axes, yticks=[])

plt.tight_layout(h_pad=3)
plotListHillaryLabel