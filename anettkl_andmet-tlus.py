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
df = pd.read_csv("../input/kc_house_data.csv")
df
df.info()

len(df)
print("Kõrgeim hind", df["price"].max(), "eurot")

print("Minimaalne hind:", df["price"].min(), "eurot")

df.groupby("yr_built").aggregate({"condition": [ "min", "max", "mean"],

                                      "grade" : [ "min", "max", "mean"]})
df.yr_built.plot.hist(bins=11, grid=False, rwidth=0.95, color="r", alpha = 0.3);
df.plot.scatter("price", "sqft_living", alpha=0.6, color = "turquoise");
