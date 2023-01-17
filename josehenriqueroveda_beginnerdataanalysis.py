import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np
df_total = pd.read_csv('../input/credit-for-agriculture-in-brazil/TotalCredit-Brazil.csv')

df_agro = pd.read_csv('../input/credit-for-agriculture-in-brazil/CreditToAgro-Brazil.csv')

df_share = pd.read_csv('../input/credit-for-agriculture-in-brazil/ShareOfTotalCredit-Brazil.csv')

df_world = pd.read_csv('../input/credit-for-agriculture-in-brazil/CreditToAgro-World.csv')
df_total.drop(['Domain Code','Domain','Area Code','Area','Element Code','Element','Item Code','Item','Year Code','Unit','Flag','Flag Description'],inplace=True,axis=1)

df_agro.drop(['Domain Code','Domain','Area Code','Area','Element Code','Element','Item Code','Item','Year Code','Unit','Flag','Flag Description'],inplace=True,axis=1)

df_share.drop(['Domain Code','Domain','Area Code','Area','Element Code','Element','Item Code','Item','Year Code','Unit','Flag','Flag Description'],inplace=True,axis=1)

df_world.drop(['Domain Code','Domain','Area Code','Area','Element Code','Element','Item Code','Item','Year Code','Unit','Flag','Flag Description'],inplace=True,axis=1)
sns.set(style='darkgrid', font_scale=1.1)

plt.figure(figsize=(18,6))



g = sns.pointplot(x=df_total.Year, y=df_total.Value, color='darkolivegreen', scale=0.8)

g.set_xticklabels(labels=df_total['Year'], fontsize=12, rotation=0)

g.set_xlabel(xlabel='Year', fontsize=16)

g.set_ylabel(ylabel='Millions US$', fontsize=16)

g.set_title(label='Total credit to economy in Brazil', fontsize=20)

plt.ticklabel_format(style='plain', axis='y')

plt.show()
plt.figure(figsize=(18,6))



g1 = sns.pointplot(x=df_agro.Year, y=df_agro.Value, color='yellowgreen', scale=0.8)

g1.set_xticklabels(labels=df_agro['Year'], fontsize=12, rotation=0)

g1.set_xlabel(xlabel='Year', fontsize=16)

g1.set_ylabel(ylabel='Millions US$', fontsize=16)

g1.set_title(label='Credit to agriculture in Brazil', fontsize=20)

plt.ticklabel_format(style='plain', axis='y')

plt.show()
df_credit = pd.merge(df_total, df_agro, left_on='Year', right_on='Year', left_index=True)
df_credit.rename(columns={'Value_x':'Total_credit',

                          'Value_y':'Agro_credit'}, 

                 inplace=True)

df_credit
plt.figure(figsize=(18,6))



plt.title('Total credit x Credit to agriculture - Brazil From 2012 to 2018')



sns.lineplot(data=df_credit['Total_credit'], label="Total credit")

sns.lineplot(data=df_credit['Agro_credit'], label="Credit to agriculture")

plt.ticklabel_format(style='plain', axis='y')

plt.xlabel('Year')

plt.ylabel('Millions US$')



plt.show()
x = np.arange(7)

fig, ax = plt.subplots()

plt.bar(x, df_share.Value)

plt.xticks(x, df_credit.Year)

plt.title('Credit for agriculture in relation to the total')

plt.xlabel('Year')

plt.ylabel('(%)')

fig.tight_layout()

fig.patch.set_facecolor('white')

plt.show()
df_share_world = pd.read_csv('../input/credit-for-agriculture-in-brazil/ShareOfTotalCredit-World.csv')
df_share_world.drop(['Domain Code','Domain','Area Code','Element Code','Element','Item Code','Item','Year Code','Unit','Flag','Flag Description'],inplace=True,axis=1)
df_share_world.head()
df_share_mean = df_share_world.groupby('Area', as_index=False)['Value'].mean()
df_share_sorted = df_share_mean.sort_values(by=['Value'], ascending=False)
plt.figure(figsize=(18,6))



g2 = sns.barplot(x=df_share_sorted.Area[:10], y=df_share_sorted.Value)

g2.set_xticklabels(labels=df_share_sorted['Area'], fontsize=12, rotation=0)

g2.set_xlabel(xlabel='Country', fontsize=16)

g2.set_ylabel(ylabel='% of credit to agriculture', fontsize=16)

g2.set_title(label='Countries with the highest % of credit for agriculture (2012-2018)', fontsize=20)

plt.ticklabel_format(style='plain', axis='y')

plt.show()