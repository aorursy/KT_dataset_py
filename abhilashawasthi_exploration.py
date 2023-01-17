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
cities = pd.read_csv("../input/cities_r2.csv")

print("Shape of the dataset: ", cities.shape)
cities.head()
# Columns/Fields in the dataset

cities.columns
# child_sex_ratio

print("Mean of child_sex_ratio of top 500 Indian Cities: ", cities.child_sex_ratio.mean())

print("Median of child_sex_ratio of top 500 Indian Cities: ", cities.child_sex_ratio.median())

print("Std Deviation of child_sex_ratio of top 500 Indian Cities: ", cities.child_sex_ratio.std())