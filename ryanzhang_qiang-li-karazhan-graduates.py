# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

plt.rcParams["figure.figsize"] = (16, 9)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read data, utf-8 encoding won't work

cards = pd.read_csv("../input/cards.csv", encoding='iso-8859-1')
# peak into the cards

cards.head()
# get the counts by set, class, type

card_counts = cards.groupby(["set", "playerClass", "type"]).size().unstack().fillna(0)
# do some house keeping for the cards added in Karazhan expension

Kara_card_counts = card_counts.loc["KARA", :].stack().reset_index()

Kara_card_counts.rename(index = str, columns = {0: "count"}, inplace = True)
# plot number of cards by type by class

g = sns.factorplot(

    x = "type", 

    y = "count", 

    data = Kara_card_counts,

    row = "playerClass", 

    kind = "bar",

    sharey = False

)

g.set_xticklabels(rotation = 90)
card_counts