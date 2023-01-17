# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

sns.set(style="whitegrid", palette="muted")



# Load the example iris dataset

iris = sns.load_dataset("iri")



# "Melt" the dataset to "long-form" or "tidy" representation

iris = pd.melt(iris, "species", var_name="measurement")



# Draw a categorical scatterplot to show each observation

sns.swarmplot(x="measurement", y="value", hue="species",

              palette=["r", "c", "y"], data=iris)