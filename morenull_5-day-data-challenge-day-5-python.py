import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats



cereal = pd.read_csv("../input/80-cereals/cereal.csv")

cereal.describe()
scipy.stats.chisquare(f_obs= cereal["fat"].value_counts())
scipy.stats.chisquare(f_obs= cereal["fiber"].value_counts())
acrossTab = pd.crosstab( cereal["fat"], cereal["fiber"])

acrossTab
scipy.stats.chi2_contingency(acrossTab)