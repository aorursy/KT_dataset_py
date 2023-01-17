import pandas as pd
df = pd.read_csv('../input/filtered-wdidata/filtered_WDIData.csv')
df.head()
df = df.drop(['Unnamed: 0', 'Country Name', 'Country Code'], axis=1)
df = pd.melt(df, id_vars=['Indicator Name', 'Indicator Code'])
df.head()
df[df['Indicator Code'] == 'EG.CFT.ACCS.ZS']
df.shape
df = df.dropna(axis=0, how='any')
df.shape
str(85728-33075) + ' null rows'
df.groupby('Indicator Name').size().sort_values(ascending=False)
df = df.rename(index=str, columns={"variable": "Year", 'value' : 'Value'})
df.to_csv('reshaped_US_WDIData.csv')