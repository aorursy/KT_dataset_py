import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



print(os.listdir("../input"))

sns.set(rc={'figure.figsize':(10, 8)});

df = pd.read_csv('../input/churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv')

df.head(10)









df.tail(10)

df.head(10).T
df.info()
df.describe().T
df.describe(include=['object','bool'])
df['churn'].value_counts(normalize=True)



df['churn'].value_counts().plot(kind='pie',label='churn')

plt.legend()

plt.title('Распределение оттока клиентов');
df['total minutes'] = df['total day minutes'] + df['total eve minutes'] + ['total night minutes'] + ['total intl minutes']

df.head()
df.pivot_table(values=['total intl calls'],index= ['international plan'], aggfunc='mean')
df.groupby('international plan')['total intl calls'].mean().plot(kind='bar') 

plt.ylabel('total intl calls') 

plt.show();
_, axes=plt.subplots(1,2,sharey=True,figsize=(16,6))

sns.countplot(x = 'international plan', hue = 'churn', data=df, ax=axes[0]);

sns.countplot(x = 'voice mail plan', hue = 'churn', data=df, ax=axes[1]);

pd.crosstab(df['churn'],df['customer service calls'],margins=True)

sns.countplot(x='customer service calls', hue='churn',data=df)
from scipy.stats import pearsonr, spearmanr, kendalltau

r = pearsonr(df['account length'], df['customer service calls'])

print('Pearson correlation:', r[0], 'p-value:', r[1])