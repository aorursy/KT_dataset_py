# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as scp

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

dataset=pd.read_csv("../input/Health_AnimalBites.csv")

#dataset.describe



observed=dataset['color'].value_counts()

#print(len(observed))

expected=dataset['BreedIDDesc'].value_counts()

#print(len(expected)

contingency=pd.crosstab(dataset['color'],dataset['BreedIDDesc'])



scp.chi2_contingency(contingency)



fig,axes=plt.subplots(ncols=2,figsize=(20,10))

#fig.subplots_adjust(wspace=1)

dataset.head(10)



sns.countplot(x='color',data=dataset.iloc[1:10,:],ax=axes[1])

axes[1].set_title("FREQUENCY DISTRIBUTION OF DOGS BASED ON COLOR")

axes[1].set_xlabel("DOG'S COLOR")

sns.countplot(x='GenderIDDesc',data=dataset,ax=axes[0])

axes[0].set_title("FREQUENCY DISTRIBUTION OF DOGS BASED ON GENDER")

axes[0].set_xlabel("GENDER")