import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()



%config InlineBackend.figure_format = 'svg'
df = pd.read_csv("../input/telecom-churn/telecom_churn.csv")
df.head()
features = ['total day minutes','total intl calls']

df[features].hist(figsize=(10,4));
df[features].plot(kind='density', subplots=True, layout=(1,2), sharex=False, figsize=(10,4))
sns.distplot(df['total intl calls'])
sns.boxplot(x='total intl calls', data=df)
_, axes = plt.subplots(1,2, sharey=True, figsize=(6,4))

sns.boxplot(data=df['total intl calls'], ax=axes[0])

sns.violinplot(data=df['total intl calls'], ax=axes[1])
df[features].describe()
df['churn'].value_counts()
_, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))



sns.countplot(x='churn', data=df, ax=axes[0])

sns.countplot(x='customer service calls', data=df, ax=axes[1])
numerical = list(set(df.columns)-set(['state', 'international plan', 'voice mail plan', 

                      'area code', 'churn', 'customer service calls']))
numerical
corr_matrix = df[numerical].corr()

sns.heatmap(corr_matrix)