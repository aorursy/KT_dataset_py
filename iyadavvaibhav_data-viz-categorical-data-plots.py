import seaborn as sns
import pandas as pd
import numpy as np
%matplotlib inline
tips = pd.read_csv('../input/tips.csv')
tips.head()
sns.barplot(x='sex',y='total_bill',data=tips) # shows mean or avg for category
sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std) # shows std dev for two
sns.countplot(x='sex',data=tips) # shows number of entries and not sum
sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.sum) # shows sum of values
sns.boxplot(x='day',y='total_bill',data=tips,palette='rainbow')
sns.boxplot(x='day',y='total_bill',data=tips,palette='rainbow', hue='smoker')
sns.violinplot(x='day',y='total_bill', data=tips)
sns.violinplot(x='day',y='total_bill', data=tips, hue='smoker')
sns.violinplot(x="day", y="total_bill", data=tips,hue='smoker',split=True,palette='Set1')
sns.stripplot(x='day', y='total_bill', data=tips)
sns.stripplot(x='day', y='total_bill', data=tips, jitter=True)
sns.stripplot(x='day', y='total_bill', data=tips, jitter=True, hue='sex')
sns.swarmplot(x="day", y="total_bill", data=tips)
sns.factorplot(x='sex',y='total_bill',data=tips,kind='bar')

