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
df = pd.read_csv("../input/Pokemon.csv")

df
df.info()
print("Ridade arv tabelis: ", df.shape[0])

print("Veergude arv tabelis: ", df.shape[1])
df[["Name", "HP", "Attack", "Defense", "Total"]].sort_values("Total", ascending=False).head(10)
putukad = df[df["Type 1"] == "Bug"]

mürgised_putukad = putukad[putukad["Type 2"] == "Poison"]

mürgised_putukad
df["Attack"].plot.hist(rwidth=0.95);
df.plot.scatter("HP","Defense",alpha=0.2)
df.groupby("Type 1").agg({"Attack": ["mean", "median"],"Defense" : ["median"]});