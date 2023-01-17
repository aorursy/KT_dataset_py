# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
plt.rc('figure', figsize=(15, 5))
data = pd.read_csv('../input/student-mat.csv')

rows, columns = data.shape

print(f"The data set has {rows} rows and {columns} columns.")

data.head()
alco = data.loc[:, ['Dalc', 'Walc', 'G1', 'G2', 'G3']]
summary = alco.describe().T

summary
summary[['mean', '50%']].plot.bar();
sns.boxplot(data=alco);
alco = alco.drop(['G1', 'G2'], axis=1)
for column in alco.columns:

    print("----- ", column, " -----")

    print(sorted(alco[column].unique()), '\n\n')
fig, axes = plt.subplots(ncols=2, sharey=True)

for i, column in enumerate(alco.columns[:2]):

    sns.countplot(alco[column], ax=axes[i]);
sns.countplot(alco['G3']);
sns.distplot(alco['G3'], bins=range(0,21), kde=True);
alco['Alc'] = alco[['Dalc', 'Walc']].mean(axis=1)
alco_corr = alco.corr()

plt.figure(figsize=(5,5))

sns.heatmap(alco_corr, annot=True, fmt='.2f',

            vmax=1, vmin=-1, center=0,

            mask=np.triu(alco_corr), cmap='coolwarm')

plt.xticks(rotation=90)

plt.yticks(rotation=0);
joint_dist = pd.crosstab(alco.Alc, alco.G3)

sns.heatmap(joint_dist, annot=True, cbar=False);
alco['grp_G3'] = pd.cut(alco.G3, bins=[-1, 5, 10, 15, 21], 

                labels=['Poor', 'Fair', 'Good', 'Excellent'])

sns.countplot(alco.grp_G3);
grp_Alc = pd.cut(alco.Alc, bins=np.arange(0, 6), labels=np.arange(1, 6))

sns.countplot(grp_Alc);
grp_G3_Alc = (pd.crosstab(alco.grp_G3, grp_Alc)

              .reindex(['Poor', 'Fair', 'Good', 'Excellent']))

sns.heatmap(grp_G3_Alc, annot=True)

plt.yticks(rotation=0);