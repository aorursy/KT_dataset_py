import pandas as pd
dados = 'kc_house_data.csv'
df = pd.read_csv(dados, sep=',', header=0)
df.head()
df.shape
df.info()
df.columns
df['bedrooms'].describe()
df['bedrooms'].unique()
df['bathrooms'].mean()
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
df = pd.read_csv(dados, sep=',', header=0)
df.head()
df.isnull().sum()
df['bedrooms'].fillna(1, inplace=True)
df.head()
df['bedrooms'].isnull().sum()
df.loc[df['bathrooms']>1]
df['bedrooms'].describe()
df['id'].loc[(df['bedrooms']>=4)]
df['id'].loc[(df['bedrooms']>=4)].count()
df.sort_values(by='price', ascending=False)
df.pivot_table('id', index='bedrooms', aggfunc = 'count', margins=True)
pd.value_counts(df['bedrooms'])
df['comodos'] = df['bedrooms'] + df['bathrooms']
df.head()
%matplotlib notebook
df['price'].hist(bins=40, color='red')
df.plot(kind='scatter', x='bathrooms', y='price', title = 'Banheiros x Preços', color='violet')
df.to_excel("Nova_planilha_housing.xlsx", index=False)