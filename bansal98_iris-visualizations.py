# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

iris = pd.read_csv("../input/Iris.csv")



# Any results you write to the current directory are saved as output.
iris.head()
## Let's see how many flowers are present for each species.

iris["Species"].value_counts()
iris.plot(x="SepalLengthCm", y="SepalWidthCm", kind="scatter")