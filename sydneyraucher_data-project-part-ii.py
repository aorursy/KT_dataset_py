# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files unWhat grade are you in?,How many times do you go across in a week?







# Any results you write to the current directory are saved as output.
import pandas as pd

data_project = pd.read_csv("../input/Data Project - Part II - Sheet1 (1).csv")
data_project.head()
data_project[["What grade are you in?", "How many times do you go across in a week?"]].corr()
from scipy import stats

pearson_coef, p_value = stats.pearsonr(data_project["What grade are you in?"], data_project["How many times do you go across in a week?"])



print(pearson_coef)



print(p_value)
import seaborn as sns

sns.boxplot(x="What grade are you in?", y="How many times do you go across in a week?", data=data_project, palette="Purples")
