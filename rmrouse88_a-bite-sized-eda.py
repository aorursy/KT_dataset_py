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
import numpy as np

import pandas as pd

from pandas import Series,DataFrame
import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns
frame = pd.read_csv("../input/Health_AnimalBites.csv")
frame.head()
by_species = frame.groupby("SpeciesIDDesc").size().to_frame().reset_index().rename(columns= {0:"Count"}).sort_values("Count",ascending = False)
print(by_species)
frame["bite_date"].sort_values(ascending = False).head()
frame["year"] = frame["bite_date"].str.split("-").str[0] 

# accesses the string split vectorized text method, then the first element of each list produced which

# should be the year
frame = frame[pd.notnull(frame["year"])] # recraft the frame to exclude those incidents missing a recorded date
frame["year"] = frame["year"].astype(np.int64) # convert the years to int
frame = frame[frame["year"] < 2018] # now recraft the frame by throwing out all incidents that occured AFTER 2018
by_year = frame.groupby('year').size().to_frame().rename(columns = {0:"Count"})

print(by_year.head())
len(frame["victim_zip"].unique())
by_zip = frame.groupby("victim_zip").size().to_frame().rename(columns={0:"Count"}).sort_values("Count",ascending=False)
targets = by_zip[by_zip["Count"] > 25] # find all zip codes where more than 25 bites have been reported

print(targets.head()) # check the list
fig,ax = plt.subplots(figsize=(12,12))

sns.heatmap(frame[frame["victim_zip"].isin(targets.index)].groupby(["victim_zip","year"]).size().unstack(),ax=ax,annot=True)
frame[(frame["SpeciesIDDesc"] == "DOG") | (frame["SpeciesIDDesc"] == "CAT")].head() # filter to grab just the observations containing dogs and ca