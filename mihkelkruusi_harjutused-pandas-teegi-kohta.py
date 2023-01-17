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
df = pd.read_csv("../input/cwurData.csv")
df
df[df["country"] == "Estonia"]
df.groupby("country")["quality_of_education"].mean()
pd.DataFrame({"keskmine_hariduse_kvaliteet" : (df.groupby("country")["quality_of_education"].mean())}).sort_values("keskmine_hariduse_kvaliteet", ascending=False)
df["country"].value_counts()

df[df["year"] == 2015]["country"].value_counts()