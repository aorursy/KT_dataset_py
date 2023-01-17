# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting library



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/summer.csv")

usa_df = df.loc[df["Country"] == "USA"]

gd = pd.DataFrame({"tally" : usa_df.groupby(["Year", "Medal"]).size()}).reset_index()

pivoted = gd.pivot(index="Year", columns="Medal", values="tally")

ax = pivoted[["Gold", "Silver", "Bronze"]].plot(kind="bar", title="USA Medal comparison by year", figsize=(12, 6))

ax.set_xlabel("Year", fontsize=10)

ax.set_ylabel("Medal Count", fontsize=10)

plt.show()