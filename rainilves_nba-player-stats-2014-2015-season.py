# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

pd.set_option('display.max_rows', 20)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/players_stats.csv")
df
df.info()
(df[["Name", "Games Played", "MIN", "PTS"]]

 .sort_values("PTS", ascending=False))
df.groupby("Games Played").aggregate({"MIN": ["sum", "mean",],

                                      "PTS" : ["sum", "mean"]})
df.Age.plot.hist(rwidth=0.95,color= "c");
df.plot.scatter("Games Played", "PTS", alpha=0.3, color= "m");