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

df = pd.read_csv("../input/2015.csv")

# Any results you write to the current directory are saved as output.
df
arv = 1

for i in df["Country"]:

    if "Estonia" in i:

        print("Eesti on " + str(arv) + ". kohal 2015. aasta statistika kohaselt." )

        break

    arv += 1
keskmine_rahulolu_hinne = df["Happiness Score"].mean()

print("Keskmine rahulolu hinne:", df["Happiness Score"].mean())
df["Happiness Score"].describe()
keskmine_rahulolu_hinne = df["Happiness Score"].mean()

üle_keskmise = df[df["Happiness Score"] > keskmine_rahulolu_hinne]
üle_keskmise
üle_keskmise = pd.DataFrame({"Region" : üle_keskmise["Region"],

             "Happiness_score" : üle_keskmise["Happiness Score"],

                            "Country" : üle_keskmise["Country"]})

üle_keskmise
üle_keskmise.groupby(["Region"])["Happiness_score"].mean()
üle_keskmise.Happiness_score.plot.hist(bins=20, grid=True, rwidth=0.8);
df.plot.scatter("Happiness Score", "Health (Life Expectancy)", alpha=0.2);
df.plot.scatter("Happiness Score", "Trust (Government Corruption)", alpha=0.2);