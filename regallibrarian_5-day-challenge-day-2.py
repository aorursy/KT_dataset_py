# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Reading in data set into Pandas dataframe

df = pd.read_csv("../input/cereal.csv")
#Examine first 5 entries of data

df.head()
#Summary of numeric data 

df.describe().transpose()
#check to see completeness of data

df.isnull().values.any(), df.shape
#Examining Sodium

df["sodium"].head()
#Histogram for sodium

sns.distplot(df["sodium"],kde=False).set_title("Sodium in Cereal")