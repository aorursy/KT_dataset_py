import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting
import seaborn as sns # more plotting

plt.style.use('seaborn-colorblind') # plotting style
import os

# Listing all datasets
pathFiles = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pathFile = os.path.join(dirname, filename)
        pathFiles.append(pathFile)
        print(pathFile)
# Importing users dataset
dfUser = pd.read_csv('/kaggle/input/argodatathon2019/users.csv', delimiter=',')
dfUser.dataframeName = 'users.csv'
nRow, nCol = dfUser.shape
print(f'There are {nRow} rows and {nCol} columns')
# Viewing the first 3 rows
dfUser.head(3)
# DataFrame details
dfUser.describe()
D = dfUser['company'].value_counts()
D
fig, ax = plt.subplots()

ax.bar(range(len(D)), D.values, align='center')
plt.xticks(range(len(D)), D.index, rotation=15)

ax.set_title('Company by Employee')
ax.set_ylabel('Number of Employees')
ax.set_xlabel('Company')

plt.show()
alphaDegree = 0.5
palette = sns.color_palette()
fig, axs = plt.subplots(len(D.index), len(D.index), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0, 'wspace': 0})
fig.suptitle("Employees' gender x Companies")
genders = dfUser['gender'].unique()

for i, company in enumerate(D.index):
    for j, company2 in enumerate(D.index):
        tmp = dfUser.query('company=="%s"' % company)
        tmp2 = dfUser.query('company=="%s"' % company2)
        tmpValues = tmp['gender'].value_counts()
        tmp2Values = tmp2['gender'].value_counts()
        axs[i, j].bar(tmpValues.index, tmpValues.values, alpha=alphaDegree, color=palette[i])
        axs[i, j].bar(tmp2Values.index, tmp2Values.values, alpha=alphaDegree, color=palette[j])
        axs[i, j].tick_params(axis="x", labelsize=8, rotation=25)
        # Vertical names
        if i==0:
            axs[i, j].set_title(D.index[j], fontsize=8)
plt.show()
alphaDegree = 0.5
palette = sns.color_palette()
fig, axs = plt.subplots(len(D.index), len(D.index), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0, 'wspace': 0})
fig.suptitle("Employees' ages x Companies")


for i, company in enumerate(D.index):
    for j, company2 in enumerate(D.index):
        tmp = dfUser.query('company=="%s"' % company)
        tmp2 = dfUser.query('company=="%s"' % company2)
        axs[i, j].hist(tmp['age'], alpha=alphaDegree, color=palette[i])
        axs[i, j].hist(tmp2['age'], alpha=alphaDegree, color=palette[j])
        # Vertical names
        if i==0:
            axs[i, j].set_title(D.index[j], fontsize=8)

dfUser.groupby(['company'])['age'].describe(percentiles=[])
