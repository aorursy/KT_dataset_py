import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings  

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/2004-2019.tsv', sep="\t")

data['DATA FINAL'] = pd.to_datetime(data['DATA FINAL'], format = '%Y-%m-%d')

data['date'] = pd.to_datetime(data['DATA FINAL'].dt.year.map(str) + '-' + data['DATA FINAL'].dt.month.map(str), format = '%Y-%m')

selected_columns = [x for x in data.columns if x not in ['Unnamed: 0', 'DATA INICIAL', 'DATA FINAL', 'MÊS', 'ANO']]

data = data[selected_columns]

data = data.replace('-', np.nan)

categorical_columns = ['REGIÃO', 'ESTADO', 'PRODUTO', 'UNIDADE DE MEDIDA', 'date']

numerical_columns = [col for col in data.columns if col not in categorical_columns]

data[numerical_columns] = data[numerical_columns].apply(pd.to_numeric)
dist_mean_prices = data.groupby(['PRODUTO', 'date', 'REGIÃO'])['PREÇO MÉDIO DISTRIBUIÇÃO'].agg('mean').unstack().reset_index().set_index('date')

dist_mean_prices_pct_change = data.groupby(['PRODUTO', 'date', 'REGIÃO'])['PREÇO MÉDIO DISTRIBUIÇÃO'].agg('mean').pct_change().unstack().reset_index().set_index('date')



fig, ax = plt.subplots(6,2, figsize=(15,20), sharex = True)

plot_pos = 1



for prod in dist_mean_prices['PRODUTO'].unique():

    

    prod_mean_price = dist_mean_prices.loc[dist_mean_prices['PRODUTO'] == prod].drop('PRODUTO', axis ='columns')

    prod_pct_change = dist_mean_prices_pct_change.loc[dist_mean_prices_pct_change['PRODUTO'] == prod].drop('PRODUTO', axis ='columns')

    

    for regiao in prod_mean_price.columns.values:

        

        act_region_prod_price = prod_mean_price[regiao]

        act_region_prod_pct_change = prod_pct_change[regiao]

        

        plt.subplot(6,2,plot_pos)

        plt.plot(act_region_prod_price, label = regiao)

        plt.xticks(rotation=90)

        plt.tight_layout()

        plt.title("MEAN DISTRIBUTION PRICE : {}".format(prod))



        plt.subplot(6,2,plot_pos+1)

        plt.plot(act_region_prod_pct_change, label = regiao)

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.xticks(rotation=90)

        plt.tight_layout()

        plt.title("% VARIATION OF MEAN DISTRIBUTION PRICE : {}".format(prod))

    

    plot_pos += 2