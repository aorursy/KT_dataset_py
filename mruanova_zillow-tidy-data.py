import pandas as pd
url = "../input/zillow-sale-counts-state/Sale_Counts_State.csv"
df_sales = pd.read_csv(url, low_memory=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df_sales.shape
df_sales.head()
df_sales.tail()
list(df_sales['RegionName'])
df = pd.melt(frame=df_sales, 
             id_vars=['RegionID','RegionName', 'SizeRank'], var_name='year-month', value_name='count')
df.head()
df.tail()
df[['year', 'month']] = df['year-month'].str.split('-', expand=True)
df.head()
df.drop(columns=['year-month'], inplace=True)
df.head()
df.tail()
df.shape