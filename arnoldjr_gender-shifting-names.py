# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/NationalNames.csv")

gb = df.groupby(['Year','Gender'], sort=False)
df.loc[:,'Freq_yearly'] = gb['Count'].transform(lambda x: 100. * x / x.sum())
df_mf = df.set_index(["Year","Name"])

df_ratio = df_mf.loc[df_mf.Gender == "F",["Freq_yearly"]].join(df_mf.loc[df_mf.Gender == "M","Freq_yearly"], 

                                                 rsuffix="_M", how="inner")

df_ratio["Ratio"] = df_ratio.Freq_yearly_M / df_ratio.Freq_yearly
df_ratio = df_ratio.reset_index().groupby('Name').filter(lambda x: len(x) > 90)
alpha = 5.0 # how much more prevalent was one name than the other, in terms of frequency

df_ratio = df_ratio.groupby("Name").filter(lambda x: x.Ratio.max() > alpha and x.Ratio.min() < (1/alpha))



df_ratio.groupby("Name").count().count()
for n, df_sub in df_ratio.groupby("Name"):

    ax = df_sub.plot.line("Year","Ratio", title=n, logy=True, legend=False, xlim=(1880,2015), ylim=(1e-3,1e2))

    ax.set_ylabel("Ratio of males to females")