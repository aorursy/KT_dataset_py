import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/telecom_churn.csv')
df.head()
print(df.shape)
print(df.columns)
df.info()
### Let us change the 'Churn' feature to 'int64'
df['churn'] = df['churn'].astype('int64')
df.head()
df.describe()
df.describe(include=['object','bool'])
df['churn'].value_counts()
df['churn'].value_counts(normalize = 'True')
df.sort_values(by='total day charge', ascending=False).head()
df.sort_values(by=['churn','total day charge'], ascending=[True, False]).head()
df['churn'].mean()
df[df['churn'] == 1].mean()
df[df['churn'] == 1]['total day minutes'].mean()
df[(df['churn'] == 0) & (df['international plan'] == 'no')]['total intl minutes'].max()
df.loc[0:5, 'state':'area code']
df.iloc[0:5, 0:3]
df[:1]
df[-1:]
df.apply(np.max)
df[df['state'].apply(lambda x : x[0] == 'W')].head()
d = {'yes' : True, 'no' : False}
df['international plan'] = df['international plan'].map(d)
df.head()
df = df.replace({'voice mail plan' : d})
df.head()
columns_to_show = ['total day minutes', 'total eve minutes', 'total night minutes']
df.groupby(['churn'])[columns_to_show].describe(percentiles=[])
df.head()
pd.crosstab(df['churn'], df['international plan'])
pd.crosstab(df['churn'], df['international plan'], normalize = True)

pd.crosstab(df['churn'],df['voice mail plan'], normalize = True)
df.pivot_table(['total day calls', 'total eve calls', 'total night calls'],
              ['area code'],
              aggfunc = 'mean')
total_calls = df['total day calls'] + df['total eve calls'] + \
                df['total night calls'] + df['total intl calls']
df.insert(loc=len(df.columns), column = 'total calls', value = total_calls)
df.head()
df['total charge'] = df['total day charge'] + df['total eve charge'] + \
                     df['total night charge'] + df['total intl charge']
df.head()
df.drop(['total charge', 'total calls'], axis=1, inplace=True)
df.head()
df.drop([1,2]).head()
pd.crosstab(df['churn'], df['international plan'], margins = True, normalize = True)
sns.countplot(x='international plan', hue='churn', data=df);
pd.crosstab(df['churn'], df['customer service calls'], margins = True)
sns.countplot(x = df['customer service calls'], hue = 'churn', data = df);
df['many service calls'] = (df['customer service calls'] > 3).astype('int')
df.head()
pd.crosstab(df['many service calls'], df['churn'], margins = True)
sns.countplot(x = df['many service calls'], hue = df['churn'], data = df);
pd.crosstab(df['many service calls'] & df['international plan'], df['churn'])
pd.crosstab(df['many service calls'] & df['international plan'], df['churn'], normalize = True)