# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #data visualization

import scipy #data analysis



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Load cereal data file (csv format)

df = pd.read_csv("../input/cereal.csv")
#Look at top 5 entries of data

df.head()
#Summarize data fields

df.describe().transpose()
#check for incomplete data

df.isnull().values.any(), df.shape
sns.distplot(df["sodium"],kde=False).set_title("Sodium in Cereal")
sns.distplot(df["sugars"],kde=False).set_title("Sugar in Cereal")
#Check to see that standard deviation is different between the two samples

np.std(df["sugars"]),np.std(df["sodium"])
scipy.stats.ttest_ind(df["sugars"],df["sodium"],equal_var=False)