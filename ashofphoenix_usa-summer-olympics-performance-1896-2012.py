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

gd = pd.DataFrame({"tally" : df.groupby(["Year", "Country"]).size()}).reset_index()

df1 = gd.loc[gd["Country"] == "USA"]

usa_tally = df1.set_index("Year")

ax = usa_tally[["tally"]].plot(kind="bar", title="USA Summer Olympics Performance 1896-2012", figsize=(10, 6))

ax.set_xlabel("Year", fontsize=12)

ax.set_ylabel("Medal Count", fontsize=12)

plt.show()