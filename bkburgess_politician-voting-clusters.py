# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn.cluster as cls

from sklearn.decomposition import PCA

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
members = pd.read_csv("../input/house-of-representatives-congress-116/house_members_116.csv", index_col="name_id")

attendance = pd.read_csv("../input/house-of-representatives-congress-116/house_rollcall_info_116.csv")

votes = pd.read_csv("../input/house-of-representatives-congress-116/house_rollcall_votes_116.csv", index_col="name_id")



def summarize(df):

    print("Columns:")

    print(df.columns)

    print("\n\n\n\n\nFirst 5 entries:")

    print(df.head())

    print("\n\n\n\n\nDescriptive Stats:")

    print(df.describe())

    print("\n\n\n\n\nMissing Info:")

    print(df.info())
summarize(members)
summarize(attendance)
summarize(votes)
attendance = attendance[['rollcall_id', 'bill_id']].dropna()



import re

p = re.compile("[HS].((R.)|(j.res.))?[0-9]*.$",re.IGNORECASE)

attendance = attendance[attendance["bill_id"].apply(lambda x: pd.notnull(p.match(x)))]

print(attendance.head())
print(votes.head())



votes = votes[attendance["rollcall_id"]] # filter out the votes on non-law things



# What are all the unique terms for voting? We need to convert strings to booleans

ls = []

for col in votes.columns:

    ls += list(pd.unique(votes[col]))

ls = pd.unique(ls)

print(ls)





# turn strings to booleans

def simplify_strings(x):

    votes_to_bool = {1: ["Aye", "Yea"], -1: ["No", "Nay"], 0:[np.nan, "Not Voting", "Present"]}

    for key, ls in votes_to_bool.items():

        if x in ls:

            return key



for col in votes.columns:

    votes[col] = votes[col].apply(simplify_strings)
votes
pca = PCA()

pca_votes = pca.fit_transform(votes)

#print(pca_votes.shape)

#print(votes.shape)

pca_votes = pd.DataFrame(pca_votes[:,:2], index = votes.index)



sns.set_style("darkgrid")

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize = (15,15))

sns.lineplot(data = np.cumsum(pca.explained_variance_ratio_), ax=ax1)

sns.scatterplot(x=pca_votes[0], y=pca_votes[1], ax=ax2)#, hue = members.loc[pca_votes.index, "current_party"])



ax1.set_title("cumulative explained variance")

ax2.set_title("politician location on first two principal components")

fig.show()

from bokeh.io import output_notebook, show

from bokeh.plotting import figure

from bokeh.models.tools import HoverTool

from bokeh.models import ColumnDataSource



output_notebook()

# Basic plot setup

p = figure(width=1200, height=600, title='Politicians in voting space')



a = ColumnDataSource.from_df(pca_votes) # dictionary of arrays {colname:np.array(colvals)}, includes index

a["0"] = a[0]

a['1'] = a[1]

del a[0]

del a[1]

a['current_party'] = np.array(members.loc[votes.index, "current_party"])

a['name']  = np.array(members.loc[votes.index, "name"])

a['state'] = np.array(members.loc[votes.index, "state"])

colors = {"Republican": "firebrick", "Democratic":"navy", "Independent": "green"}

a['color'] = np.array([colors[i] for i in members.loc[votes.index, "current_party"]])



TOOLTIPS = [

    ("name", "@name"),

    ("state", "@state"),

    ("party", "@current_party")

]

cr = p.circle(x="0", y="1", radius=0.3,

              hover_fill_color="cyan",

              fill_alpha=0.8, hover_alpha=0.8, fill_color='color',

              line_color="white", hover_line_color="white", source=a)



p.add_tools(HoverTool(tooltips=TOOLTIPS, renderers=[cr]))



show(p)