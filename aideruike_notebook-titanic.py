# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

from scipy import stats

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head(1)
train.describe()
print(train.isnull().sum())

print(test.info())
surv = train[train['Survived']==1]

nosurv = train[train['Survived']==0]

print("Survived: {}({:.2f} percent), Not Survived: {}(({:.2f} percent)), Total: {}"\

      .format(len(surv), 1.*len(surv)/len(train)*100.0, len(nosurv), 1.*len(nosurv)/len(train)*100.0, len(train)))
surv_col = "blue"

nosurv_col = "red"

plt.figure(figsize=[12,10])

plt.subplot(331)

sns.distplot(surv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=surv_col)

sns.distplot(nosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=nosurv_col, axlabel='Age')

plt.subplot(332)

sns.barplot('Sex', 'Survived', data=train)

plt.subplot(333)

sns.barplot('Pclass', 'Survived', data=train)

plt.subplot(334)

sns.barplot('Embarked', 'Survived', data=train)

plt.subplot(335)

sns.barplot('SibSp', 'Survived', data=train)

plt.subplot(336)

sns.barplot('Parch', 'Survived', data=train)

plt.subplot(337)

sns.distplot(np.log10(surv['Fare'].dropna().values+1), kde=False, color=surv_col)

sns.distplot(np.log10(nosurv['Fare'].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)



print("Median age survivors: {:.1f}, Median age non-survivers: {:.1f}"\

      .format(np.median(surv['Age'].dropna()), np.median(nosurv['Age'].dropna())))
train.groupby(['Sex'])['Survived'].groups
tab = pd.crosstab(train['SibSp'], train['Survived'])

print(tab)

dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, color=[nosurv_col,surv_col])

dummy = plt.xlabel('SibSp')

dummy = plt.ylabel('Percentage')
train[['Survived','Cabin']].dropna().head(8)
train.loc[:,['Survived','Cabin']].dropna().head(8)
train.iloc[0:2]
grouped = train.groupby('Ticket')

k = 0

for name, group in grouped:

    if (len(grouped.get_group(name)) > 1):

        print(group.loc[:,['Survived','Name']])

        k += 1

    if (k>10):

        break
stats.binom_test(x=2,n=2,p=0.5)
grouped = train.groupby('Ticket')
for name, group in grouped:

    print (name)
grouped.get_group('113773')
grouped.groups
train.corr()
train['Survived']
np.corrcoef([train['Survived']],[train['Age']])
stats.pearsonr(train['Survived'],train['Age'])
train['Survived'].dropna()
train['Survived'].dropna().values
np.size(train['Age'].dropna())

a=train[train['Age'].notnull()]['Survived'].values

b=train['Age'].dropna().values
np.size(a)
train['Age'].isnull()
np.size(train)
stats.pearsonr(a,b)