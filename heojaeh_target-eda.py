import numpy as np 

import pandas as pd 



# graph

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')



import os

print(os.listdir("../input"))
label =  pd.read_csv('../input/train_label.csv')

print('label.shape: ',label.shape)
plt.figure(figsize=(7,5))

sns.kdeplot(label.survival_time, shade=True)

plt.title('Survival Time Distribution')

plt.show()
label.survival_time.describe()
survived = label.survival_time.apply(lambda x: 'Survived' if x == 64 else 'Not Survived')

survived_freq = survived.value_counts()

survived_ratio = (survived_freq / survived_freq.sum()).round(2)

sns.countplot(survived)

plt.title('Survived Ratio')

plt.xlabel('')

for i in range(survived_ratio.shape[0]):

    plt.text(i-0.1, survived_freq.iloc[i]-2000, survived_ratio.iloc[i], color='white', size=15)
plt.figure(figsize=(7,5))

sns.kdeplot(label.amount_spent, shade=True)

plt.title('Amount Spent Distribution')

plt.show()
label.amount_spent.describe()
f, ax = plt.subplots(1,2, figsize=(15,5))

sns.scatterplot(x='survival_time', y='amount_spent', data=label, ax=ax[0])

ax[0].set_title('Correlation Survival time and Average Amount Spent')

ax[1].set_ylabel('Average Amount Spent')

ax[1].set_xlabel('Survival Time')



# Add total amount spent = survival time * amount spent(average)

label['total_amount_spent'] = label.survival_time * label.amount_spent

sns.scatterplot(x='survival_time', y='total_amount_spent', data=label, ax=ax[1])

ax[1].set_title('Correlation Survival Time and Total Amount Spent')

ax[1].set_ylabel('Total Amount Spent')

ax[1].set_xlabel('Survival Time')

plt.show()