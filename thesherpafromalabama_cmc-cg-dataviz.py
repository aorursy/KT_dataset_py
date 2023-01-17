# Run this cell to pip install statnnot

!pip install statannot
# Imports and initializations



import pandas as pd

import seaborn as sns



# for stattanot, make sure you 

from statannot import add_stat_annotation

import matplotlib.pyplot as plt

import matplotlib

from scipy import stats



exchange_df = pd.read_csv('../input/coingecko-vs-coinmarketcap-data/May_CMC_CG_Combo.csv')



matplotlib.__version__
exchange_df['Bad_Pairs'] = exchange_df['Unknown_Pairs'] + exchange_df['Yellow_Pairs'] + exchange_df['Red_Pairs']
# Get numeric df

numeric_df = exchange_df._get_numeric_data()

# Clean out estimated reserves and regulatory compliance

numeric_df = numeric_df.drop(columns = ['Estimated_Reserves', 'Regulatory_Compliance'])



# Get correlation matrix

corr = numeric_df.corr()



fig, ax = plt.subplots(figsize=(10,10))  

sns.heatmap(corr, annot = True, linewidths=.5, ax = ax)
num_cols = numeric_df.columns

fig, ax = plt.subplots(3,4,figsize=(22,10))

for i, col in enumerate(num_cols):

    plt.subplot(3,5,i+1)

    plt.xlabel(col, fontsize=9)

    sns.kdeplot(numeric_df[col].values, bw=0.5,label='Train')

    # sns.kdeplot(raw_test[col].values, bw=0.5,label='Test')

   

plt.show() 
# Set styleee

sns.set_style("dark")



# Get boxplot

plot = sns.violinplot(exchange_df['Websocket'], exchange_df['CMC Liquidity'])

plot = sns.stripplot(x='Websocket', y='CMC Liquidity', data=exchange_df, color="purple", jitter=0.3, size=3.0)

# Get statistical info

add_stat_annotation(plot, data=exchange_df, x=exchange_df['Websocket'], y=exchange_df['CMC Liquidity'],

                                   box_pairs=[("Available", "Not")],

                                   test='t-test_ind', text_format='star',

                                   loc='outside', verbose=2)
plot = sns.violinplot(exchange_df['Trading_via_API'], exchange_df['CMC Liquidity'])

plot = sns.stripplot(x='Trading_via_API', y='CMC Liquidity', data=exchange_df, color="purple", jitter=0.3, size=3.0)

add_stat_annotation(plot, data=exchange_df, x=exchange_df['Trading_via_API'], y=exchange_df['CMC Liquidity'],

                                   box_pairs=[("Available", "Not")],

                                   test='t-test_ind', text_format='star',

                                   loc='outside', verbose=2)
plot = sns.violinplot(exchange_df['Websocket'], exchange_df['Liquidity'])

plot = sns.stripplot(x='Websocket', y='Liquidity', data=exchange_df, color="purple", jitter=0.3, size=3.0)

add_stat_annotation(plot, data=exchange_df, x=exchange_df['Websocket'], y=exchange_df['Liquidity'],

                                   box_pairs=[("Available", "Not")],

                                   test='t-test_ind', text_format='star',

                                   loc='outside', verbose=2)
order = [" Low", " Medium", " High"]

plot = sns.boxplot(exchange_df['Sanctions'], exchange_df['CMC Liquidity'], order = order)

plot = sns.stripplot(x='Sanctions', y='CMC Liquidity', data=exchange_df, color="purple", jitter=0.3, size=3.0 ,order = order)

add_stat_annotation(plot, data=exchange_df, x=exchange_df['Sanctions'], y=exchange_df['CMC Liquidity'], order = order,

                                   box_pairs=[(" Low", " High"), (" Low", " Medium")],

                                   test='Mann-Whitney', text_format='star',

                                   loc='outside', verbose=2)

order = [' High', ' Medium', ' Low']

plot = sns.boxplot(exchange_df['Negative_News'], exchange_df['CMC Liquidity'], order = order)

plot = sns.stripplot(x='Negative_News', y='CMC Liquidity', data=exchange_df, color="purple", jitter=0.3, size=3.0 ,order = order)

add_stat_annotation(plot, data=exchange_df, x=exchange_df['Negative_News'], y=exchange_df['CMC Liquidity'], order = order,

                                   box_pairs=[(" Low", " High"), (" Low", " Medium"), (' High', ' Medium')],

                                   test='Mann-Whitney', text_format='star',

                                   loc='outside', verbose=1)
plot = sns.scatterplot(data = exchange_df, x = 'Bad_Pairs', y = 'CMC Liquidity', hue = 'Trust_Score', x_jitter=2.0)
plot = sns.scatterplot(data = exchange_df, x = 'Red_Pairs', y = 'CMC Liquidity', hue = 'Trust_Score', x_jitter=2.0)
plot = sns.scatterplot(data = exchange_df, x = 'Bad_Pairs', y = 'BidAsk Spread', hue = 'Trust_Score', x_jitter=2.0)
def r2(x, y):

    return stats.pearsonr(x, y)[0] ** 2



# sns.jointplot(data = exchange_df, x = 'Scale', y = 'CMC Liquidity', kind = 'reg', stat_func=r2)

sns.regplot(data = exchange_df, x = 'Scale', y = 'CMC Liquidity')
plot = sns.jointplot(data = exchange_df, x = 'Trust_Score', y = 'CMC Liquidity', kind = 'reg', stat_func=r2)