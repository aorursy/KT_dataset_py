import seaborn as sns
data = sns.load_dataset('penguins')
data.shape
data.describe(include = 'all')
data.info()
data.isna().sum()
data[data.isnull().any(axis=1)]
sns.pairplot(data)