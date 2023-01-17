import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from ggplot import *

import seaborn as sns

sns.set_style("whitegrid")

%matplotlib inline



np.seterr(invalid='ignore')

pd.set_option('max_colwidth', 20)



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv') 

df = train.append(test)



print('Dimensions of the data:', train.shape, test.shape, df.shape)
# A quick look

cat_vars = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Cabin']

text_vars = ['Name', 'Ticked']

num_vars = ['Age', 'Fare']

train.head(5)
fig, axs = plt.subplots(nrows=5, figsize=(8,20), sharex=False)

for i in range(5):

    sns.countplot(x=cat_vars[i], data=train, hue='Survived', ax=axs[i])
sns.factorplot(x="Pclass", y="Age", hue="Survived", data=train, kind="box")
sns.factorplot(x="Pclass", y="Fare", hue="Survived", data=train[train.Fare < 200], kind="box")
## The same can be found after the log transform

train.loc[:,'logFare'] = np.log1p(train.Fare)

sns.factorplot(x="Pclass", y="logFare", hue="Survived", data=train, kind="box")
# This would be some sort of "univariate" way of using the function

sns.pointplot(x="Pclass", y="Survived", hue="Pclass", data=train)
# Comparing Embarked and Pclass by the proportion of Survived

sns.pointplot(x="Embarked", y="Survived", hue="Pclass", data=train)
sns.pointplot(x="SibSp", y="Survived", hue="SibSp", data=train)
sns.pointplot(x="Parch", y="Survived", hue="Parch", data=train)
sns.lmplot(x="Fare", y="Age", hue="Survived", data=train, fit_reg=False)