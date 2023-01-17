# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

import numpy as np

import pandas as pd 



%matplotlib inline

pd.set_option('display.max_rows', 30)



df = pd.read_csv("../input/igen.csv")

df
print('Maksimaalne skoor: ', df['score'].max())

print('Minimaalne skoor: ',df['score'].min())

print('keskmine skoor: ',df['score'].mean())

# Kui paju mänge on ühes või teises žanris?

df["genre"].value_counts()
# Kui palju mänge on platformidele loodud?

df["platform"].value_counts()
# Kõige paremad/halvemad mängud.

(df[["title", "platform", "genre", "score",'release_year']]

 .sort_values("score", ascending=False))
#Keskmised hinnangud aastate kohta.

df.plot.scatter( "release_year","score", alpha=.02);
# Hinnangute arvukus

df.score.plot.hist(bins=10, grid=False, rwidth=.95); 
#Aasta parim mäng                                                                    #suht noname mängud

df.groupby(["release_year"])["score",'title'].max()