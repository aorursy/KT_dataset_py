import seaborn as sns
df = sns.load_dataset("tips")
df.head()
df.dtypes

df.corr()

sns.heatmap(df.corr())
sns.jointplot(x='tip', y='total_bill', data = df, kind='hex')
sns.jointplot(x='tip', y='total_bill', data=df, kind='reg') 
sns.pairplot(df)
sns.pairplot(df, hue = 'sex')
df['smoker'].value_counts()
sns.pairplot(df, hue = 'smoker')
sns.distplot(df['tip'])