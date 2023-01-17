# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn # data visualization
%matplotlib inline
sn.set (color_codes = True, style="white")
import matplotlib.pyplot as ml #data visualization
import warnings
warnings.filterwarnings("ignore")

print ("Environment works!!!")
# Let's import our data

events = pd.read_csv("../input/athlete_events.csv", sep = ",", header = 0)
# Let's take a look at the shape and size of our data

events.shape
# Let's remove the "ID" column and make a new dataset called "events_clean

events_clean = events.drop(["ID"], axis=1)
# Let's take a look at our data

events_clean.head(10)
# Let's replace the words "NaN", "Bronze", "Silver" and "Gold" with numerical values that we can plot

z = {"Bronze" : 1, "Silver" : 2, "Gold" : 3}
events_clean["Medal"] = events_clean["Medal"].map(z) 

# Let's replace the letters "F" and "M"  with numerical values that we can plot

z = {"F" : 1, "M" : 0}
events_clean["Sex"] = events_clean["Sex"].map(z) 
# Let's see what the "Medals" column looks like now

events_clean.head(10)
ml.subplots(figsize=(15,15))
sn.boxplot (data=events_clean)
ml.subplots(figsize=(15,15))
sn.violinplot(x="NOC", y="Medal", data=events_clean)
# Let's take a look at the heatmap

ml.subplots(figsize=(10,10))
co = events_clean.corr()
sn.heatmap(co, annot=True, linewidths=1.0)