import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

file_path = '../input/crime-dataset/crime.csv'

 

print('Setup Complete')
data = pd.read_csv('../input/crime-dataset/crime.csv', encoding='iso-8859-1')

data = data.dropna()
data.columns
plt.figure(figsize=(6, 50))

ax = sns.countplot(y = 'REPORTING_AREA', data = data, saturation = 0.75)

ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right")

plt.show()
sns.set(style='darkgrid')

plt.figure(figsize=(16, 6))

ax = sns.countplot(x = 'OFFENSE_CODE_GROUP', saturation = 0.75, dodge = True, data = data, order = data['OFFENSE_CODE_GROUP'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.title('Count Plot for Offense Code Group')

plt.tight_layout()

plt.show()

data.OFFENSE_CODE_GROUP.value_counts().iloc[:]
sns.set(style='darkgrid')

plt.figure(figsize=(25, 50))

ax = sns.countplot(y = 'REPORTING_AREA', saturation = 0.5, dodge = True, data = data, order = data['REPORTING_AREA'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.tight_layout()

plt.show()

data.REPORTING_AREA.value_counts().iloc[:]