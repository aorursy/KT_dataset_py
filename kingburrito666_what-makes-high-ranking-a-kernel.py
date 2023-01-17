import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# what makes datasets popular?

datakern = pd.read_csv('../input/top-kernels.csv')
datakern.info()
datakern.describe()
datakern.columns
sns.set(style="whitegrid")

sns.pairplot(datakern, hue="Language",

             x_vars=["Upvotes", "Comments"],

             y_vars=["Upvotes", "visuals"],

             palette="husl",

             size=3)
sns.set(style="whitegrid")



# Draw a categorical scatterplot to show each observation

sns.swarmplot(palette="husl", x="isNotebook", y="Upvotes", hue="Language", data=datakern)
sns.set(style='whitegrid', palette='husl')

# Draw using seaborn

sns.jointplot('Comments','Upvotes',

          data=datakern, ylim=(0,400),

          xlim=(0,200), kind='reg')
lenttl = []

for i in datakern['Name']:

    lenttl.append(len(i))

    

up_to_title = pd.DataFrame(columns=('Title length', 'Upvotes'))

up_to_title['Title length'] = lenttl

up_to_title['Upvotes'] = datakern['Upvotes']
sns.set(style='whitegrid', palette='husl')

sns.jointplot('Title length','Upvotes',

          data=up_to_title, kind='reg')
sns.set(style='whitegrid')

# Draw using seaborn

sns.lmplot('Comments','Upvotes',

          data=datakern, palette='husl',

          hue='Language', fit_reg=True)