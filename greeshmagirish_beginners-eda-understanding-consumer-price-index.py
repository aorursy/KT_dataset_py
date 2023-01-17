import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style='whitegrid')

sns.set_context('talk')
cpi_df = pd.read_csv("../input/cpi_2013_Oct_2019_india.csv")

cpi_df = cpi_df.drop('Unnamed: 0', axis=1)
cpi_df.Description.unique()
df1 = cpi_df[cpi_df['Description'] == 'General Index (All Groups)']
fig,ax = plt.subplots(1,1, figsize=(20,10))

sns.lineplot(df1['Year'], df1['Rural'], ci=None)

plt.ylabel('CPI Index Current Series (Base 2012) ')

plt.title('Rural General Index From January 2013 - October2019', fontsize=15)
fig,ax = plt.subplots(1,1, figsize=(20,10))

sns.lineplot(df1['Year'], df1['Urban'], ci=None)

plt.ylabel('CPI Index Current Series (Base 2012) ')

plt.title('Urban General Index From January 2013 - October2019', fontsize=15)
df1 = cpi_df[cpi_df['Description'] == 'Fuel and light']
fig,ax = plt.subplots(1,1, figsize=(20,10))

sns.lineplot(df1['Year'], df1['Rural'], ci=None)

plt.ylabel('CPI Index Current Series (Base 2012) ')

plt.title('Rural Index for Fuel and Light From January 2013 - October2019', fontsize=15)
fig,ax = plt.subplots(1,1, figsize=(20,10))

sns.lineplot(df1['Year'], df1['Urban'], ci=None)

plt.ylabel('CPI Index Current Series (Base 2012) ')

plt.title('Urban Index for Fuel and Light From January 2013 - October2019', fontsize=15)
df1 = cpi_df[cpi_df['Description'] == 'Clothing and footwear']
fig,ax = plt.subplots(1,1, figsize=(20,10))

sns.lineplot(df1['Year'], df1['Rural'], ci=None)

plt.ylabel('CPI Index Current Series (Base 2012) ')

plt.title('Rural Index for Clothing and Footwear From January 2013 - October2019', fontsize=15)
fig,ax = plt.subplots(1,1, figsize=(20,10))

sns.lineplot(df1['Year'], df1['Urban'], ci=None)

plt.ylabel('CPI Index Current Series (Base 2012) ')

plt.title('Urban Index for Clothing and Footwear From January 2013 - October2019', fontsize=15)
df1 = cpi_df[cpi_df['Description'] == 'Housing']
fig,ax = plt.subplots(1,1, figsize=(20,10))

sns.lineplot(df1['Year'], df1['Urban'], ci=None)

plt.ylabel('CPI Index Current Series (Base 2012) ')

plt.title('Urban Index for Housing From January 2013 - October2019', fontsize=15)
df1 = cpi_df[cpi_df['Description'] == 'Pan; tobacco; and intoxicants']
fig,ax = plt.subplots(1,1, figsize=(20,10))

sns.lineplot(df1['Year'], df1['Rural'], ci=None)

plt.ylabel('CPI Index Current Series (Base 2012) ')

plt.title('Rural Index for Pan, Tobacco and Intoxicants From January 2013 - October2019', fontsize=15)
fig,ax = plt.subplots(1,1, figsize=(20,10))

sns.lineplot(df1['Year'], df1['Urban'], ci=None)

plt.ylabel('CPI Index Current Series (Base 2012) ')

plt.title('Urban Index for Pan, Tobacco and Intoxicants From January 2013 - October2019', fontsize=15)
df1 = cpi_df[cpi_df['Description'] == 'Food and beverages']
fig,ax = plt.subplots(1,1, figsize=(20,10))

sns.lineplot(df1['Year'], df1['Rural'], ci=None)

plt.ylabel('CPI Index Current Series (Base 2012) ')

plt.title('Rural Index for Food and Beverages From January 2013 - October2019', fontsize=15)
fig,ax = plt.subplots(1,1, figsize=(20,10))

sns.lineplot(df1['Year'], df1['Rural'], ci=None)

plt.ylabel('CPI Index Current Series (Base 2012) ')

plt.title('Urban Index for Food and Beverages From January 2013 - October2019', fontsize=15)
df1 = cpi_df[cpi_df['Description'] == 'Consumer Food Price Index']
fig,ax = plt.subplots(1,1, figsize=(20,10))

sns.lineplot(df1['Year'], df1['Rural'], ci=None)

plt.ylabel('CPI Index Current Series (Base 2012) ')

plt.title('Rural Consumer Food Price Index From January 2013 - October2019', fontsize=15)
fig,ax = plt.subplots(1,1, figsize=(20,10))

sns.lineplot(df1['Year'], df1['Urban'], ci=None)

plt.ylabel('CPI Index Current Series (Base 2012) ')

plt.title('Urban Consumer Food Price Index From January 2013 - October2019', fontsize=15)