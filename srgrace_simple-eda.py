# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt



df = pd.read_csv("../input/clash_wiki_dataset.csv")
df.head()
df.info()
df.describe()
df.columns
import seaborn as sns
f, ax = plt.subplots(figsize = (18, 18))

sns.heatmap(df.corr(), annot = True, linewidths = .5, fmt = '.1f', ax = ax)
df.Cost.plot(kind = 'hist', bins = 50)
df.tail()
plt.figure(figsize=(12,10))

plt.ylim(0, 275)

sns.swarmplot(x="Type", y="Cost", data=df, hue="Card", dodge=True, size=7)

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.);
plt.figure(figsize=(15,15))

plt.ylim(0, 275)

sns.boxplot(data = df);