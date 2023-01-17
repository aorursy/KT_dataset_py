# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # for visualisations



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# first we read data from the csv 

data = pd.read_csv("../input/Health_AnimalBites.csv")
# To view the first few rows of data from the data set 

data.head()
# We put the contents of the "WhereBittenIDDesc" in a variable so we can plot it

x = data["WhereBittenIDDesc"]



# To get a summary of the column

data["WhereBittenIDDesc"].describe()
# Plotting the graph 

sns.countplot(x).set_title("WHERE A VICTIM WAS BITTEN")