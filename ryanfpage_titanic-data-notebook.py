# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# Add a reader -- checkout pandas

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64},)

# Replacing missing ages with median

#train["Age"][np.isnan(train["Age"])] = np.median(train["Age"])

median_age = train['Age'].dropna().median() # not sure about this... but cant think of a better measure



if len(train.Age[ train.Age.isnull() ]) > 0:

    train.loc[ (train.Age.isnull()), 'Age'] = -99

    

train.head(5)

    
#train.loc[(train['Survived']==1), 'Survived'] = "Survived"

#train.loc[(train['Survived']==0), 'Survived'] = "Died"





plt.figure()

sns.pairplot(data=train[["Fare","Survived","Age","Parch","SibSp","Pclass"]],

             hue="Survived", dropna=True)
#Get max and min

maxAge = train.Age.max()

minAge = train.Age.min()

if minAge < 0:

    minAge = 0

nbins = int(math.ceil( maxAge-minAge))

bins = np.arange(0,nbins+1,1)

fig, ax = plt.subplots(1,1)

ageColSur = train[train.Survived==1].Age.as_matrix()

ageColDied = train[train.Survived==0].Age.as_matrix()



ax.hist(ageColSur[~np.isnan(ageColSur)], bins=bins, align='left', normed=True, alpha=0.5, label='Sur')

ax.hist(ageColDied[~np.isnan(ageColDied)], bins=bins, align='left', normed=True, alpha=0.5, label='Died')

plt.legend(loc='upper right')