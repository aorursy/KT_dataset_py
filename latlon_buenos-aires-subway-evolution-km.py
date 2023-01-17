# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))





ll = pd.read_csv("../input/lines.csv")

tl = pd.read_csv("../input/track_lines.csv")

df = pd.read_csv("../input/tracks.csv")



lines = ll[ll["system_id"]==254]

lines.head(3)

track_lines = tl[tl["line_id"].isin(lines["id"].values)]

tracks = df[df["id"].isin(track_lines["section_id"].values)]



kms=[]

for y in range(1910, 2019):

    kms.append(tracks[tracks["opening"] >= y].length.sum()/1000)



d = dict([['year',list(range(1910, 2019))],['km', list(reversed(kms))]])

data = pd.DataFrame.from_dict(d, orient="columns")

    

g = sns.factorplot(x="year",y="km", data=data, height=7, scale=.5)

g.set_xticklabels(rotation=45)





for label in g.ax.xaxis.get_ticklabels():

    if not (label.get_text().endswith('5') or label.get_text().endswith('0')):

        label.set_visible(False)

    