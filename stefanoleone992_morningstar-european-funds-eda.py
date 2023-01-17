import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

pd.set_option('display.max_columns', None)

pd.options.mode.chained_assignment = None
column_names = ['fund_name', 'isin', 'morningstar_category', 'morningstar_benchmark', 'dividend_frequency', 'morningstar_rating',

                'morningstar_analyst_rating', 'morningstar_risk_rating', 'morningstar_performance_rating', 'nav_per_share_currency',

                'nav_per_share', 'class_size_currency', 'class_size', 'fund_return_ytd', 'fund_return_2018', 'fund_return_2017',

                'fund_return_2016', 'fund_return_2015', 'fund_return_2014', 'fund_return_2013', 'fund_return_2012', 'fund_return_2011',

                'fund_return_2010', 'trailing_return_3years', 'trailing_return_5years', 'equity_style', 'equity_style_score',

                'equity_size', 'equity_size_score', 'roa', 'roe', 'roic', 'bond_interest_rate_sensitivity', 'bond_credit_quality',

                'average_coupon_rate', 'average_credit_quality', 'modified_duration', 'effective_maturity', 'asset_stock', 'asset_bond',

                'asset_cash', 'asset_other', 'holdings_number_stock', 'holdings_number_bonds', 'ongoing_cost', 'management_fees',

                'sustainability_rank', 'esg_score', 'environmental_score', 'social_score', 'governance_score', 'controversy_score',

                'sustainability_score']

df_mutual_funds = pd.read_csv("../input/european-funds-dataset-from-morningstar/Morningstar - European Mutual Funds.csv",

                              usecols=column_names)

df_etfs = pd.read_csv("../input/european-funds-dataset-from-morningstar/Morningstar - European ETFs.csv",

                      usecols=column_names)
numerical_columns = ['morningstar_rating', 'morningstar_risk_rating', 'morningstar_performance_rating', 'ongoing_cost',

                     'management_fees', 'fund_return_ytd', 'trailing_return_3years', 'trailing_return_5years',

                     'sustainability_rank', 'esg_score', 'environmental_score', 'social_score', 'governance_score',

                     'controversy_score', 'sustainability_score']

df_mutual_funds = df_mutual_funds.reset_index().set_index('isin')

df_mutual_funds_numerical = df_mutual_funds[numerical_columns]

print('N. elements in Mutual Funds original dataframe: ' + str(len(df_mutual_funds_numerical)))

df_mutual_funds_numerical = df_mutual_funds_numerical.dropna().apply(pd.to_numeric)

print('N. elements in Mutual Funds dataframe after NAs are dropped: ' + str(len(df_mutual_funds_numerical)))



df_etfs = df_etfs.reset_index().set_index('isin')

df_etfs_numerical = df_etfs[numerical_columns]

print('N. elements in ETFs original dataframe: ' + str(len(df_etfs_numerical)))

df_etfs_numerical = df_etfs_numerical.dropna().apply(pd.to_numeric)

print('N. elements in ETFs dataframe after NAs are dropped: ' + str(len(df_etfs_numerical)))



# list of mutual funds

mutual_funds_list = list(df_mutual_funds_numerical.index.values)

df_mutual_funds = df_mutual_funds[df_mutual_funds.index.isin(mutual_funds_list)]



# list of ETFs

etfs_list = list(df_etfs_numerical.index.values)

df_etfs = df_etfs[df_etfs.index.isin(etfs_list)]
df_mutual_funds.drop('index', axis=1, inplace=True)

df_mutual_funds.head()
df_etfs.drop('index', axis=1, inplace=True)

df_etfs.head()
f, axes = plt.subplots(3, 1, figsize=(20, 20), sharex=False)

sns.distplot(df_mutual_funds_numerical[(df_mutual_funds_numerical['fund_return_ytd'] >= -30) & (df_mutual_funds_numerical['fund_return_ytd'] <= 30)]['fund_return_ytd'],

             ax=axes[0]).set_title("Mutual Funds - YearToDate return")

axes[0].set(xlabel='Return', ylabel='Distribution', xlim=(-30, 30))

sns.distplot(df_mutual_funds_numerical[(df_mutual_funds_numerical['trailing_return_3years'] >= -30) & (df_mutual_funds_numerical['trailing_return_3years'] <= 30)]['trailing_return_3years'],

             ax=axes[1]).set_title("Mutual Funds - 3Years return")

axes[1].set(xlabel='Return', ylabel='Distribution', xlim=(-30, 30))

sns.distplot(df_mutual_funds_numerical[(df_mutual_funds_numerical['trailing_return_5years'] >= -30) & (df_mutual_funds_numerical['trailing_return_5years'] <= 30)]['trailing_return_5years'],

             ax=axes[2]).set_title("Mutual Funds - 5Years return")

axes[2].set(xlabel='Return', ylabel='Distribution', xlim=(-30, 30))
f, axes = plt.subplots(3, 1, figsize=(20, 20), sharex=False)

sns.distplot(df_etfs_numerical[(df_etfs_numerical['fund_return_ytd'] >= -30) & (df_etfs_numerical['fund_return_ytd'] <= 30)]['fund_return_ytd'],

             ax=axes[0]).set_title("ETFs - YearToDate return")

axes[0].set(xlabel='Return', ylabel='Distribution', xlim=(-30, 30))

sns.distplot(df_etfs_numerical[(df_etfs_numerical['trailing_return_3years'] >= -30) & (df_etfs_numerical['trailing_return_3years'] <= 30)]['trailing_return_3years'],

             ax=axes[1]).set_title("ETFs - 3Years return")

axes[1].set(xlabel='Return', ylabel='Distribution', xlim=(-30, 30))

sns.distplot(df_etfs_numerical[(df_etfs_numerical['trailing_return_5years'] >= -30) & (df_etfs_numerical['trailing_return_5years'] <= 30)]['trailing_return_5years'],

             ax=axes[2]).set_title("ETFs - 5Years return")

axes[2].set(xlabel='Return', ylabel='Distribution', xlim=(-30, 30))
morningstar_mutual_funds_category_df = df_mutual_funds[df_mutual_funds['morningstar_category'].notnull()]



plt.figure(figsize=(15, 8))

mutual_funds_categories = morningstar_mutual_funds_category_df['morningstar_category'].value_counts()[:10]

sns.barplot(mutual_funds_categories.values, mutual_funds_categories.index, )

for i, v in enumerate(mutual_funds_categories.values):

    plt.text(0.8,i,v,color='k',fontsize=12)

plt.xlabel('Count', fontsize=12)

plt.ylabel('Category name', fontsize=12)

plt.title("Mutual Funds - Distribution of Morningstar categories", fontsize=16)
morningstar_etfs_category_df = df_etfs[df_etfs['morningstar_category'].notnull()]



plt.figure(figsize=(15, 8))

etfs_categories = morningstar_etfs_category_df['morningstar_category'].value_counts()[:10]

sns.barplot(etfs_categories.values, etfs_categories.index, )

for i, v in enumerate(etfs_categories.values):

    plt.text(0.8,i,v,color='k',fontsize=12)

plt.xlabel('Count', fontsize=12)

plt.ylabel('Category name', fontsize=12)

plt.title("ETFs - Distribution of Morningstar categories", fontsize=16)
fund_categories_list = ['Asia ex-Japan Equity', 'China Equity', 'EUR Corporate Bond', 'Eurozone Large-Cap Equity',

                        'Germany Large-Cap Equity', 'Global Emerging Markets Equity', 'Global Large-Cap Blend Equity',

                        'Japan Large-Cap Equity', 'Switzerland Large-Cap Equity', 'UK Large-Cap Equity',

                        'US Large-Cap Blend Equity', 'USD Corporate Bond']

mutual_funds_category_df = df_mutual_funds.loc[df_mutual_funds['morningstar_category'].isin(fund_categories_list)].sort_values(by='morningstar_category', ascending=True)

plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.violinplot(x=mutual_funds_category_df['morningstar_category'], y=mutual_funds_category_df['esg_score'], palette='magma')

ax.set_xlabel(xlabel='Morningstar categories', fontsize=12)

ax.set_ylabel(ylabel='ESG score', fontsize=12)

ax.set_title(label='Distribution of ESG score of Mutual Funds from different Morningstar categories', fontsize=16)

plt.xticks(rotation=60)

plt.show()
etfs_category_df = df_etfs.loc[df_etfs['morningstar_category'].isin(fund_categories_list)].sort_values(by='morningstar_category', ascending=True)

plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.violinplot(x=etfs_category_df['morningstar_category'], y=etfs_category_df['esg_score'], palette='inferno')

ax.set_xlabel(xlabel='Morningstar categories', fontsize=12)

ax.set_ylabel(ylabel='ESG score', fontsize=12)

ax.set_title(label='Distribution of ESG score of ETFs from different Morningstar categories', fontsize=16)

plt.xticks(rotation=60)

plt.show()
plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.barplot(x=mutual_funds_category_df['morningstar_category'], y=mutual_funds_category_df['environmental_score'], palette='magma')

ax.set_xlabel(xlabel='Morningstar categories', fontsize=12)

ax.set_ylabel(ylabel='Environmental score', fontsize=12)

ax.set_title(label='Distribution of Environmental score of Mutual Funds from different Morningstar categories', fontsize=16)

plt.xticks(rotation=60)

plt.show()
plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.barplot(x=etfs_category_df['morningstar_category'], y=etfs_category_df['environmental_score'], palette='inferno')

ax.set_xlabel(xlabel='Morningstar categories', fontsize=12)

ax.set_ylabel(ylabel='Environmental score', fontsize=12)

ax.set_title(label='Distribution of Environmental score of ETFs from different Morningstar categories', fontsize=16)

plt.xticks(rotation=60)

plt.show()
plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.boxplot(x=mutual_funds_category_df['morningstar_category'], y=mutual_funds_category_df['social_score'], palette='magma')

ax.set_xlabel(xlabel='Morningstar categories', fontsize=12)

ax.set_ylabel(ylabel='Social score', fontsize=12)

ax.set_title(label='Distribution of Social score of Mutual Funds from different Morningstar categories', fontsize=16)

plt.xticks(rotation=60)

plt.show()
plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.boxplot(x=etfs_category_df['morningstar_category'], y=etfs_category_df['social_score'], palette='inferno')

ax.set_xlabel(xlabel='Morningstar categories', fontsize=12)

ax.set_ylabel(ylabel='Social score', fontsize=12)

ax.set_title(label='Distribution of Social score of ETFs from different Morningstar categories', fontsize=16)

plt.xticks(rotation=60)

plt.show()
plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.boxenplot(x=mutual_funds_category_df['morningstar_category'], y=mutual_funds_category_df['governance_score'], palette='magma')

ax.set_xlabel(xlabel='Morningstar categories', fontsize=12)

ax.set_ylabel(ylabel='Governance score', fontsize=12)

ax.set_title(label='Distribution of Governance score of Mutual Funds from different Morningstar categories', fontsize=16)

plt.xticks(rotation=60)

plt.show()
plt.rcParams['figure.figsize'] = (20, 10)

ax = sns.boxenplot(x=etfs_category_df['morningstar_category'], y=etfs_category_df['governance_score'], palette='inferno')

ax.set_xlabel(xlabel='Morningstar categories', fontsize=12)

ax.set_ylabel(ylabel='Governance score', fontsize=12)

ax.set_title(label='Distribution of Governance score of ETFs from different Morningstar categories', fontsize=16)

plt.xticks(rotation=60)

plt.show()