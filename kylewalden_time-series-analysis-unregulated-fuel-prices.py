import pandas as pd

import seaborn as sns

df = pd.read_excel('../input/Fuel_Transactions_by_Oil_Company.xlsx')

df.info()
df['Brand Description'].value_counts()
df['Fuel Type'].value_counts() #two types of fuel
df['PPL'] = df['Amount'] / df['Quantity'] #lets bring in the price per litre

#remove price regulated fuel types in South Africa from the dataframe, in this case petrol

df = df[df['Fuel Type'] != 'Petrol']
import pandas_profiling

df.profile_report()
plot = sns.lineplot(x='Transaction Date', y='PPL', data=df, hue='Brand Description') #plot for interest
df = df[df.duplicated(subset=['Merchant Town'], keep=False)] #address merchant towns

df = df.sort_values(by=['Brand Description', 'Transaction Date']) #address time

df['Date Diff'] = df['Transaction Date'].diff() 

pd.to_datetime(df['Date Diff'])

df['total_days_td'] = df['Date Diff'] / pd.to_timedelta(1, unit='D')

df = df[df['total_days_td'] >= 0] #remove brand description group days difference overlap

df = df.groupby('Brand Description').filter(lambda g: (g.total_days_td < 30).all()) #filter brands not used within 30 days

df['Brand Description'].unique() #lets check the sample
df = df[df['Brand Description'] != 'OTHER/ANDER'] #Remove unidentified fuel brands

df = df[df['Brand Description'] != 'BRENT OIL'] #Remove unidentified fuel brands

df = df[df['Brand Description'] != 'SERVICES AND MAINTENANCE'] #Remove unidentified brands

df['Brand Description'].unique() #our resultant sample groups
plot = sns.lineplot(x='Transaction Date', y='PPL', data=df, hue='Brand Description')
df['MA'] = df.groupby('Brand Description')['PPL'].transform(lambda x: x.rolling(30, 1).mean())

plot = sns.lineplot(x='Month', y='MA', data=df, hue='Brand Description')
plot = sns.barplot(x='Brand Description', y='PPL', data=df)
df.groupby('Brand Description')['PPL'].mean().sort_values()