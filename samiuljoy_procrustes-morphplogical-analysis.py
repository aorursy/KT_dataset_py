# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()
sns.set_color_codes()

from scipy import stats

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will l all files under the input directory
print('Data paths:\n...........\n')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print('\n...............\nImport complete!')
# Any results you write to the current directory are saved as output.
# reading data
data = pd.read_csv('/kaggle/input/procrustes_landmarks.csv')
print('data successfully loaded!')
data.head(n=6)
data.tail()
data.min().head()
data.max().head()
data.describe()
data['ontogeny'].value_counts().head()
data.describe()
data['heteroblasty']
data['species'].value_counts().head()
data['plant'].value_counts().head()
sns.lmplot(x='ontogeny', y='heteroblasty', data=data, fit_reg=False)
sns.lmplot(x='x1', y='y1', data=data, fit_reg=True)
sns.scatterplot(x = data['heteroblasty'], y = data['ontogeny'], color='#778899', marker='o')
# ontogeny vs heteriblasty graph
plt.plot(data.loc[0:20, ['heteroblasty']], data.loc[0:20, ['heteroblasty']], linestyle=':', marker='o', c='#778899')
plt.title('heteroblasty vs ontogeny graph')
plt.xlabel('heteroblasty')
plt.ylabel('ontogeny')
plt.show()
sns.kdeplot(data['total'].value_counts(), shade=True, color='#778899')
plt.legend()
ax = sns.countplot(x="species", data=data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7)
plt.show()
data.head()
data['heteroblasty'].value_counts().plot(kind='bar', grid=True, color='#778899', figsize=(10,5))
plt.title('Heteroblasty value counts')
plt.show()
data['species'].value_counts().plot(kind='bar', grid=True, color='#778899', figsize=(10,5))
plt.title('species value counts')
plt.show()
plt.fill(data['x1'], data['y1'], data['x2'], data['y2'], data['x3'], data['y3'], data['x4'], data['y4'], data['x5'], data['y5'], data['x6'], data['y6'], data['x7'], data['y7'], data['x8'], data['y8'], data['x9'], data['y9'], data['x10'], data['y10'], data['x11'], data['y11'], data['x12'], data['y12'], data['x13'], data['y13'], data['x14'], data['y14'], data['x15'], data['y15'], c='#778899', alpha=0.5, linewidth=0.5)# Plot some data on the axes.
plt.title('x1/y1 values')
plt.xlabel('x1 values')
plt.ylabel('y1 values')
plt.show()
actinata_x = data.loc[0, ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15']]
actinata_y = data.loc[0, ['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12', 'y13', 'y14', 'y15']]
plt.plot(actinata_x, actinata_y, c='#778899')
print(data.loc[0, ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15']])
plant0_xvalues = data.loc[0, ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15']]
actinata_x = []

for values in plant0_xvalues:
    actinata_x.append(values)

print(actinata_x)
plant0_yvalues = data.loc[0, ['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12', 'y13', 'y14', 'y15']]
actinata_y = []

for values in plant0_yvalues:
    actinata_y.append(values)

print(actinata_y)
plt.plot(actinata_x, actinata_y)
at_least_10 = data[ data['total'] >= 10 ]
at_least_10['species'].value_counts()[0:8]

tricuspis = at_least_10[ at_least_10['species']=="tricuspis" ]
misera = at_least_10[ at_least_10['species']=="misera" ]
miersii = at_least_10[ at_least_10['species']=="miersii" ]
suberosa = at_least_10[ at_least_10['species']=="suberosa" ]
racemosa = at_least_10[ at_least_10['species']=="racemosa" ]
sidifolia = at_least_10[ at_least_10['species']=="sidifolia" ]
coriacea = at_least_10[ at_least_10['species']=="coriacea" ]
foetida = at_least_10[ at_least_10['species']=="foetida" ]

veins_x = tricuspis.iloc[:, [5]+[17]+[7]+[21]+[9]+[25]+[11]+[29]+[13]+[33]+[15]+[5] ]
veins_y = tricuspis.iloc[:, [6]+[18]+[8]+[22]+[10]+[26]+[12]+[30]+[14]+[34]+[16]+[6] ]

blade_x = tricuspis.iloc[:, [5]+[17]+[19]+[21]+[23]+[25]+[27]+[29]+[31]+[33]+[15]+[5] ]
blade_y = tricuspis.iloc[:, [6]+[18]+[20]+[22]+[24]+[26]+[28]+[30]+[32]+[34]+[16]+[6] ]


i = 6

plt.fill(blade_x.iloc[i], -blade_y.iloc[i], c='#778899', alpha=0.5)
plt.fill(veins_x.iloc[i], -veins_y.iloc[i], c='#10101c')
plt.plot(veins_x.T.iloc[:], -veins_y.T.iloc[:], alpha=0.05, c='k')
plt.plot(blade_x.T.iloc[:], -blade_y.T.iloc[:], alpha=0.05, c='g')
plt.axes().set_aspect('equal', 'datalim')
plt.axis('off')
plt.title('Transposing veins/blades')
