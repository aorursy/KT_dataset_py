# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #data visualization

from scipy.stats import chisquare #data analysis 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#import data set

df = pd.read_csv("../input/DigiDB_digimonlist.csv")
#Examine data for a quick look over

df.head()
#Examine Numerical data

df.describe().transpose()
sns.countplot(df["Stage"]).set_title("Distribution of Digimon based on Stage")
sns.countplot(df["Attribute"]).set_title("Distribution of Digimon based on Attribute")
scipy.stats.chisquare(df["Attribute"].value_counts())
scipy.stats.chisquare(df["Type"].value_counts())