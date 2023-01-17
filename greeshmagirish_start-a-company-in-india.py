import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('darkgrid')

sns.set_context('notebook')
private_df = pd.read_csv('/kaggle/input/private-sector-indicators-for-india-1.csv')

private_df = private_df.drop(private_df.index[0])

private_df.Value = private_df.Value.astype(float)

private_df.Year = private_df.Year.astype(int)
df = private_df[private_df['Indicator Name'] == 'Trade in services (% of GDP)'].filter(items=['Year', 'Value'])

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Value', data=df)

plt.fill_between(df.Year.values, df.Value.values)

plt.xticks(rotation=40)

plt.title('Trade in Services', fontsize=20);
df1 = private_df[private_df['Indicator Name'] == 'Insurance and financial services (% of service imports, BoP)'].filter(items=['Year', 'Value'])

df2 = private_df[private_df['Indicator Name'] == 'Insurance and financial services (% of service exports, BoP)'].filter(items=['Year', 'Value'])

df1 = df1.rename(columns = {"Value": "Imports"})

df2 = df2.rename(columns = {"Value": "Exports"})

df = pd.merge(df1, df2, on='Year')

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Imports', data=df, color='blue', label='Imports')

sns.lineplot("Year", 'Exports', data=df, color='green', label='Exports')

plt.ylabel('% of Service - BoP')

plt.title('Insurance and Financial Services', fontsize=20);
df = private_df[private_df['Indicator Name'] == 'Domestic credit to private sector (% of GDP)'].filter(items=['Year', 'Value'])

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Value', data=df)

plt.fill_between(df.Year.values, df.Value.values)

plt.ylabel('% of GDP')

plt.title('Domestic Credit to Private Sector', fontsize=20);
df1 = private_df[private_df['Indicator Name'] == 'Ease of doing business score (0 = lowest performance to 100 = best performance)'].filter(items=['Year', 'Value'])

df1 = df1.rename(columns = {"Value": "Score"})

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Score', data=df1)

plt.title('Ease of Doing Business', fontsize=20);
df2 = private_df[private_df['Indicator Name'] == 'Ease of doing business index (1=most business-friendly regulations)'].filter(items=['Year', 'Value'])

df2
df1 = private_df[private_df['Indicator Name'] == 'New businesses registered (number)'].filter(items=['Year', 'Value'])

df1 = df1.rename(columns = {"Value": "Number"})

fig,ax = plt.subplots(figsize=(20,15))

sns.barplot("Year", 'Number', data=df1, palette='Set2')

plt.title('New Businesses Registered', fontsize=20);
df = private_df[private_df['Indicator Name'] == 'Average time to clear exports through customs (days)'].filter(items=['Year', 'Value'])

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Value', data=df)

plt.fill_between(df.Year.values, df.Value.values)

plt.ylabel('Days')

plt.title('Average Time to Clear Exports Through Customs', fontsize=20);
df = private_df[private_df['Indicator Name'] == 'Time to obtain an electrical connection (days)'].filter(items=['Year', 'Value'])

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Value', data=df)

plt.fill_between(df.Year.values, df.Value.values)

plt.ylabel('Days')

plt.title('Time to Obtain Electrical Connection', fontsize=20);
df = private_df[private_df['Indicator Name'] == 'Burden of customs procedure, WEF (1=extremely inefficient to 7=extremely efficient)'].filter(items=['Year', 'Value'])

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Value', data=df)

plt.ylabel('Number')

plt.title("Burden of Custom's Procedure", fontsize=20);
df = private_df[private_df['Indicator Name'] == 'Time required to register property (days)'].filter(items=['Year', 'Value'])

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Value', data=df)

plt.ylabel('Days')

plt.title("Time Required to Register Property", fontsize=20);
df = private_df[private_df['Indicator Name'] == 'Start-up procedures to register a business, female (number)'].filter(items=['Year', 'Value'])

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Value', data=df)

plt.ylabel('Number')

plt.title('Start-Up Procedures to Register a Business for Women', fontsize=20);
df = private_df[private_df['Indicator Name'] == 'Start-up procedures to register a business, male (number)'].filter(items=['Year', 'Value'])

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Value', data=df)

plt.ylabel('Number')

plt.title('Start-Up Procedures to Register a Business for Men', fontsize=20);
df = private_df[private_df['Indicator Name'] == 'Cost of business start-up procedures, female (% of GNI per capita)'].filter(items=['Year', 'Value'])

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Value', data=df)

plt.fill_between(df.Year.values, df.Value.values)

plt.ylabel('% of GNI per capita')

plt.title('Cost of Business Start-Up Procedures for Women', fontsize=20);
df = private_df[private_df['Indicator Name'] == 'Cost of business start-up procedures, male (% of GNI per capita)'].filter(items=['Year', 'Value'])

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Value', data=df)

plt.fill_between(df.Year.values, df.Value.values)

plt.ylabel('% of GNI per capita')

plt.title('Cost of Business Start-Up Procedures for Men', fontsize=20);
df = private_df[private_df['Indicator Name'] == 'Time to prepare and pay taxes (hours)'].filter(items=['Year', 'Value'])

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Value', data=df)

plt.ylabel('Hours')

plt.title('Time to Prepare/Pay Taxes', fontsize=20);
df = private_df[private_df['Indicator Name'] == 'Labor tax and contributions (% of commercial profits)'].filter(items=['Year', 'Value'])

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Value', data=df)

plt.ylabel('% of Commercial Profits')

plt.title('Labor Tax and Contributions', fontsize=20);
df = private_df[private_df['Indicator Name'] == 'Profit tax (% of commercial profits)'].filter(items=['Year', 'Value'])

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Value', data=df)

plt.ylabel('% of Commercial Profits')

plt.title('Profit Tax', fontsize=20);
df = private_df[private_df['Indicator Name'] == 'Other taxes payable by businesses (% of commercial profits)'].filter(items=['Year', 'Value'])

fig,ax = plt.subplots(figsize=(20,15))

sns.lineplot("Year", 'Value', data=df)

plt.ylabel('% of Commercial Profits')

plt.title('Other Taxes Payable', fontsize=20);