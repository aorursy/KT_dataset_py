import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

bank_data = pd.read_csv("../input/bank-marketing-dataset/bank.csv")

bank_data.head(5)
bank_data.shape

bank_data.info()
bank_data.describe()
sns.set_style('whitegrid')

plt.figure(figsize=(14,7))

sns.barplot(x=bank_data['deposit'], y=bank_data['age'])

sns.set_style('whitegrid')

plt.figure(figsize=(14,7))

sns.barplot(x=bank_data['deposit'], y=bank_data['balance'])
sns.set_style('whitegrid')

plt.figure(figsize=(14,7))

sns.barplot(x=bank_data['deposit'], y=bank_data['campaign'])
s= (bank_data.dtypes =='object')

objectcols = list(s[s].index)

bank_data_object = bank_data[objectcols]

bank_data_object.head(5)

sns.set_style('whitegrid')

plt.figure(figsize=(14,7))

sns.countplot(bank_data['job'])
sns.set_style('whitegrid')

plt.figure(figsize=(14,7))

sns.countplot(bank_data['marital'])
sns.set_style('whitegrid')

plt.figure(figsize=(14,7))

sns.countplot(bank_data['education'])
sns.set_style('whitegrid')

plt.figure(figsize=(14,7))

sns.countplot(bank_data['loan'])
sns.set_style('whitegrid')

plt.figure(figsize=(14,7))

sns.countplot(bank_data['housing'])
plt.figure(figsize=(14,7))

#sns.heatmap(data=df, annot=True)

cor = bank_data.corr()

sns.heatmap(cor, annot=True)

plt.show()
(bank_data['deposit']=='yes').sum()
sns.set_style('whitegrid')

plt.figure(figsize=(14,7))

sns.countplot(bank_data['deposit'])