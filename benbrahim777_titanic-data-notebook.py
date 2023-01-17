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



df = pd.read_csv("../input/train.csv")

df.head(5)
#Get max and min

maxAge = df.Age.max()

minAge = df.Age.min()

nbins = int(math.ceil( maxAge-minAge))

bins = np.arange(0,nbins+11,nbins/8)

fig, ax = plt.subplots(1,1)

ageColSur = df[df.Survived==1].Age.as_matrix()

ageColDied = df[df.Survived==0].Age.as_matrix()



ax.hist(ageColSur[~np.isnan(ageColSur)], bins=bins, align='left', normed=True, alpha=0.5)

ax.hist(ageColDied[~np.isnan(ageColDied)], bins=bins, align='left', normed=True, alpha=0.5)
nfig, nax = plt.subplots(1,1)

ageColDiedSib = df[(df.Parch==1)].Survived.as_matrix()

ageColDiedNoSib = df[(df.Parch==0)].Survived.as_matrix()



nax.hist(ageColDiedSib[~np.isnan(ageColDiedSib)],  align='left', normed=True, alpha=0.5)

nax.hist(ageColDiedNoSib[~np.isnan(ageColDiedNoSib)], align='left', normed=True, alpha=0.5)


plt.figure()

df["Survived"][df["Survived"]==1] = "Survived"

df["Survived"][df["Survived"]==0] = "Died"

sns.pairplot(data=df[["Fare","Survived","Age","Parch","SibSp","Pclass"]],

             hue="Survived", dropna=True)

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )



# Replacing missing ages with median

#train["Age"][np.isnan(train["Age"])] = np.median(train["Age"])

median_age = train['Age'].dropna().median()



if len(train.Age[ train.Age.isnull() ]) > 0:

    train.loc[ (train.Age.isnull()), 'Age'] = median_age



train.loc[(train['Survived']==1), 'Survived'] = "Survived"

train.loc[(train['Survived']==0), 'Survived'] = "Died"



train["ParentsAndChildren"] = train["Parch"]

train["SiblingsAndSpouses"] = train["SibSp"]



plt.figure()

sns.pairplot(data=train[["Fare","Survived","Age","ParentsAndChildren","SiblingsAndSpouses","Pclass"]],

             hue="Survived", dropna=True)