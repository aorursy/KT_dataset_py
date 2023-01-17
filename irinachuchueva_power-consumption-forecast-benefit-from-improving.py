import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random as random



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
timestep_name = 'timestep'



# United Energy System we're going to consider (could be UES_Northwest, UES_Siberia, UES_Center)

ues = 'UES_Northwest'



# How we're going to aggreate accuracy cost, this is for yearly value

agg_pattern = '%Y'



# How many random scenarios of deviation we're going through

number_of_scenarios = 100

rub_eur_exchange = 70.8                 # I took an average exchange rate for 2017-2019, rub/euro
price = pd.read_csv('/kaggle/input/russian-wholesale-electricity-market/RU_Electricity_Market_UES_dayahead_price.csv')

price.index = price[timestep_name]

price.drop(timestep_name, axis=1, inplace=True)



indicator = pd.read_csv('/kaggle/input/russian-wholesale-electricity-market/RU_Electricity_Market_UES_intraday_price.csv')

indicator.index = indicator[timestep_name]

indicator.drop(timestep_name, axis=1, inplace=True)



# Make sure we're considering the same time range

if len(indicator) < len(price):

    dates_range = indicator.index

else:

    dates_range = price.index



price = price.loc[dates_range]

indicator = indicator.loc[dates_range]
# Let's pretend our consumer consumes 100 MWh every sigle hour

actuals = np.ones(len(price)) * 100



# Variable to write result

accuracy_benefit_agg = dict()
def get_forecast(act):



    # Create forecast as random deviation from actuals

    f_1 = act + np.random.normal(loc=0, scale=10, size=(len(act)))

    f_2 = f_1.copy()



    # Define hours when consumer buys or sells deviation higher then 1 MWh

    buy = (f_1 - act) > 1

    sell = (f_1 - act) < -1



    # Help consumer to improve its forecast by 1 MWh:

    f_2[buy] = f_1[buy] - 1         # when it buys, we reduce volume to buy

    f_2[sell] = f_1[sell] + 1       # when it sells, we reduce volume to sell



    return f_1, f_2, buy, sell
def get_profit(act, fsct, da_pr, bm_ind):



    # Buy forecast by day ahead price, use minus to point that it's cost

    cost_da = fsct * da_pr * -1

    dev = fsct - act



    cost_bm_buy = np.zeros(len(indicator))

    revenue_bm_sell = np.zeros(len(indicator))



    # Define hours when consumer buys deviation

    buy = dev < 0

    sell = np.invert(buy)



    # If consumer forecast was lower that actual

    # then it buys shortage by maximum of day ahead price and balance market indicator

    cost_bm_buy[buy] = dev[buy] * np.maximum(da_pr, bm_ind)[buy]



    # In other case when consumer forecast was higher than actual

    # it sells surplus by minimum of day ahead price and balance market indicator

    revenue_bm_sell[sell] = dev[sell] * np.minimum(da_pr, bm_ind)[sell]



    prof = cost_da + cost_bm_buy + revenue_bm_sell



    return prof
for scen in range(number_of_scenarios):



    forecast_1, forecast_2, buy_mask, sell_mask = get_forecast(actuals)



    profit_1 = get_profit(actuals, forecast_1, price[ues], indicator[ues])

    profit_2 = get_profit(actuals, forecast_2, price[ues], indicator[ues])



    accuracy_benefit = profit_2 - profit_1

    y = accuracy_benefit.groupby(pd.to_datetime(accuracy_benefit.index).strftime(agg_pattern)).sum()

    

    if len(accuracy_benefit_agg) == 0:

        accuracy_benefit_agg[ues] = y.values

    else:

        accuracy_benefit_agg[ues] = np.row_stack((accuracy_benefit_agg[ues], y.values))

    

    if scen % 20 == 0:

        print('Scenario %s' % scen)
row_index = ['scen_' + str(s) for s in range(number_of_scenarios)]

df_rub = pd.DataFrame(accuracy_benefit_agg[ues], index=row_index, columns=[y.index])


# Get yearly benefit for 2019 (as values available only till the end of September)

# df_rub['year_2019'] = np.array(df_rub['year_2019']) / 9 * 12          # Somehow the line doesn't work, who knows why?

df_rub.iloc[:, 2] = df_rub.iloc[:, 2] / 9 * 12                          # Replacement



# Change the currency

df_eur = df_rub / rub_eur_exchange

df_eur.head()
# Plot hist for entire time range

hist = dict()

width = dict()

center = dict()

for y in range(len(df_eur.columns)):

    gist_data = df_eur.iloc[:, y].values

    step = (max(gist_data) - min(gist_data)) / 15

    bins = np.arange(min(gist_data), max(gist_data), step)

    hist[y], bins = np.histogram(gist_data, bins=bins)

    width[y] = np.diff(bins)

    center[y] = (bins[:-1] + bins[1:]) / 2
fig, axs = plt.subplots(2, 2, figsize=(20,15))

axs[0, 0].bar(center[0], hist[0], align='center', width=width[0])

axs[0, 0].set_title('2017, mean = %5.0f eur' % np.mean(df_eur.iloc[:, 0].values))

axs[0, 1].bar(center[1], hist[1], align='center', width=width[1])

axs[0, 1].set_title('2018, mean = %5.0f eur' % np.mean(df_eur.iloc[:, 1].values))

axs[1, 0].bar(center[2], hist[2], align='center', width=width[2])

axs[1, 0].set_title('2019, mean = %5.0f eur' % np.mean(df_eur.iloc[:, 2].values))



for ax in axs.flat:

    ax.set(xlabel='Accuracy benefit,  eur', ylabel=' Number of scenarios')

plt.show()