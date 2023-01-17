# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from scipy import stats, integrate

from scipy.stats import ttest_ind

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
digimons = pd.read_csv("../input/DigiDB_digimonlist.csv")

print(digimons.describe(include='all'))
myplot = sns.countplot(x="Stage",data=digimons)

myplot.set_title("Digimon Stages")
myplot = sns.countplot(x="Stage",hue="Type", data=digimons)

myplot.set_title("Digimon Stages")
sns.swarmplot(data=digimons, x="Stage", y="Memory").set_title("Digimons: Stage and memory")