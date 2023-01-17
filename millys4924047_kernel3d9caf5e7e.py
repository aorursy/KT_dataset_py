# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
mushroom = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")

mushroom.head()
mushroom["class"].nunique()
mushroom.shape    
mushroom.count()
mushroom["class"].value_counts()
mushroom["class"].value_counts().plot.bar()
mushroom.describe()
pd.plotting.parallel_coordinates(mushroom, "class")
pd.plotting.parallel_coordinates(mushroom.drop("odor", axis=1), "class")
this_mushroom = mushroom[['odor', 'class', 'bruises']]

pd.plotting.parallel_coordinates(this_mushroom, "class")
mush_subs = mushroom [["class", "odor", "habitat", "cap-color"]]

pd.plotting.parallel_coordinates(mush_subs, "class")
mush_subs=mushroom[["class", "odor", "habitat", "cap-color"]]

pd.plotting.parallel_coordinates(mush_subs, "class")
import seaborn as sns

sns.pairplot(mushroom, hue="class") #data is not continuous so willnot give any sensible results
# this one doesn't do anything either, perhaps issues with non-continuous data again?



mush_subs = mushroom [["class", "odor", "habitat", "cap-color"]]

pd.plotting.andrews_curves(mush_subs, "class") 