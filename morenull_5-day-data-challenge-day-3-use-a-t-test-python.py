import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



from scipy.stats import ttest_ind



cereal = pd.read_csv("../input/cereal.csv")

cereal.describe()
ttest_ind(cereal["fat"],cereal["sugars"],equal_var= False)
ttest_ind(cereal["fat"],cereal["rating"],equal_var= False)
sns.distplot(cereal["fat"],kde= False)
sns.distplot(cereal["sugars"],kde= False)
sns.distplot(cereal["rating"],kde= False)