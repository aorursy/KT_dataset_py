# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



%matplotlib inline

pd.set_option('display.max_rows', 40)

df = pd.read_csv("../input/menu.csv")



df
df = df[df["Category"] == "Breakfast"] 

df[df["Calories"] > 600]
df[df["Calories"] == 1150]
df[df["Calories"] == 150]
df["Total Fat"].describe()
df["Protein"].describe()
df["Carbohydrates"].describe()
df["Dietary Fiber"].plot.hist();
df.plot.scatter("Calories", "Calories from Fat");
df.groupby("Calories").aggregate({"Vitamin A (% Daily Value)": ["mean"],

                                      "Vitamin C (% Daily Value)" : ["mean"],

                                      "Calcium (% Daily Value)": ["mean"],

                                      "Iron (% Daily Value)" : ["mean"]})
df[df["Vitamin C (% Daily Value)"] == 130]