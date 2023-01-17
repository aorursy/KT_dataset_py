import numpy as np

import pandas as pd



import plotly.graph_objects as go

import plotly.io as pio



pio.renderers.default = "notebook"
# pio.renderers
BASE_URL = 'https://docs.google.com/spreadsheets/d/10cihRP6XxJOSrWsASfW5C7VeHufD32TS0JKCXLoWhL8/export?format=csv&{}'



UKX_sheet = BASE_URL.format('gid=134530274')

HUKX_sheet = BASE_URL.format('gid=1614447400')

ISF_sheet = BASE_URL.format('gid=1374752918')

# Read data

df = pd.read_csv(UKX_sheet)
df.head(3)
# Rename columns

df = df.rename(columns={'Unnamed: 1': 'Open'})



# Convert to datetime

df['Date'] = pd.to_datetime(df['Date'].str[:10], format="%d/%m/%Y")



# Get datetime related fields

df['month'] = df['Date'].dt.month_name()

df['year'] = df['Date'].dt.year

df['day'] = df['Date'].dt.day

df['day_name'] = df['Date'].dt.day_name()

df['weekofyear'] = df['Date'].dt.weekofyear

df['is_quarter_end'] = df['Date'].dt.is_quarter_end

df['is_quarter_start'] = df['Date'].dt.is_quarter_start

df['quarter'] = df['Date'].dt.quarter

df['avg'] = round((df['High'] + df['Low']) / 2, 2) # $\frac{high + low}{2}$



# df = df.set_index('Date')

df.head()
LIMIT = 2000000 # £20,000 per year

FEE = 995





# From 1999 to 2019, the average inflation was 2.7%

INFLATION = 2.8





jan_2020_mean_price = df.loc[(df['year'] == 2020) & (df['month'] == 'January'), 'avg'].mean()
def get_stats(strategy_df):

    tmp = strategy_df.copy()

    

    total_shares = tmp['shares'].sum()

    amount_invested = int((LIMIT * tmp.shape[0])/100)



    value = int((total_shares * jan_2020_mean_price)/100)

    profit = value - amount_invested

    value_increase = round((profit/amount_invested)*100, 2)



    print('Total no. of Shares:', int(total_shares))

    print('Amount invested: £{:,}'.format(amount_invested))

    print('Total value of Shares in Jan 2020: £{:,}'.format(value))

    print('Total profit in {} years: £{:,}'.format(tmp.shape[0], profit))

    print('Total value increased: {}%'.format(value_increase))

    

    

    

def get_cum_shares_with_div(shares, end_quarter_price, dividend_yield=0.05):

    """

    Given an iterable shares and end of quarter prices:

    

    Calculates the cumulative shares with dividends reinvested

    using the end of quarter price.

    

    """

    result = []

    total_shares = 0



    for shares, price in zip(shares, end_quarter_price):



        total_shares += shares

        value = total_shares * price

        dividend = value * dividend_yield

        div_shares = dividend / price

        total_shares += div_shares



        result.append(total_shares)



    return result

monthly_limit = (LIMIT/12) - FEE



benchmark = (

        monthly_limit/(df

            .loc[

                df['Date'].isin(

                    df

                    .loc[(df['year'] >= 2001) & (df['year'] <= 2019), ['Date', 'year', 'month']]

                    .groupby(['year', 'month'])

                    .agg({'Date': min})

                    .reset_index()

                    .loc[:, 'Date']

                    .to_list()

                    ), 

                ['Date', 'avg']

            ]

        )    

        .set_index('Date')

    )



benchmark = benchmark.reset_index()

benchmark = (

    benchmark

    .groupby([benchmark['Date'].dt.year, benchmark['Date'].dt.quarter])

    .sum()

    .rename(columns={'avg': 'shares'})

)



benchmark.index = benchmark.index.set_names(['year', 'quarter'])

benchmark = benchmark.reset_index()
benchmark.head()
get_stats(benchmark.groupby('year').agg({'shares': 'sum'}))
quarterly_limit = (LIMIT/4) - FEE



upper_limit = (    

        (

            quarterly_limit/(df

                .loc[(df['year'] >= 2001) & (df['year'] <= 2019)]

                .groupby(['year', 'quarter'])

                .agg({'Low': min})

                .reset_index()

                .set_index('year')

                .loc[:, ['Low']]

            )

        )

        .reset_index()

        .rename(columns={'Low': 'shares'})

)

upper_limit.tail(3)
get_stats(upper_limit.groupby('year').agg({'shares': 'sum'}))
quarterly_limit = (LIMIT/4) - FEE



lower_limit = (    

        (

            quarterly_limit/(df

                .loc[(df['year'] >= 2001) & (df['year'] <= 2019)]

                .groupby(['year', 'quarter'])

                .agg({'High': max})

                .reset_index()

                .set_index('year')

                .loc[:, ['High']]

            )

        )

        .reset_index()

        .rename(columns={'High': 'shares'})

)
lower_limit.tail(3)
get_stats(lower_limit.groupby('year').agg({'shares': 'sum'}))
strategies = pd.DataFrame(index=pd.date_range(start='2001-01-01', end='2019-12-31', freq='Q'))



strategies['upper_limit'] = upper_limit['shares'].to_list()

strategies['benchmark'] = benchmark['shares'].to_list()

strategies['lower_limit'] = lower_limit['shares'].to_list()



strategies.head(3)
fig = go.Figure()



fig.add_trace(go.Scatter(

                x=strategies.index,

                y=strategies['benchmark'],

                name="benchmark",

                line_color='grey',

                opacity=0.8)

             )



fig.add_trace(go.Scatter(

                x=strategies.index,

                y=strategies['upper_limit'],

                name="upper_limit",

                line_color='deepskyblue',

                opacity=0.8)

             )



fig.add_trace(go.Scatter(

                x=strategies.index,

                y=strategies['lower_limit'],

                name="lower_limit",

                line_color='red',

                opacity=0.8)

             )





# Use date string to set xaxis range

fig.update_layout(

    title_text="No. of Shares purchased by year",

    # xaxis_range=['2016-07-01','2016-12-31'],

)





fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

                x=strategies.index,

                y=strategies['benchmark'].cumsum(),

                name="benchmark",

                line_color='grey',

                opacity=0.8)

             )



fig.add_trace(go.Scatter(

                x=strategies.index,

                y=strategies['upper_limit'].cumsum(),

                name="upper_limit",

                line_color='deepskyblue',

                opacity=0.8)

             )



fig.add_trace(go.Scatter(

                x=strategies.index,

                y=strategies['lower_limit'].cumsum(),

                name="lower_limit",

                line_color='red',

                opacity=0.8)

             )



# Use date string to set xaxis range

fig.update_layout(

    title_text="Total no. of Shares by year",

#     xaxis_range=['2016-07-01','2016-12-31'],

)





fig.show()
tmp = strategies.copy().reset_index(drop=True)

quarterly_dividend_yield = 0.01



end_quarter_dates = (

    df

    .loc[(df['year'] >= 2001) & (df['year'] <= 2019)]

    .groupby(['year', 'quarter'])

    .agg({'Date': max}).reset_index().loc[:, 'Date']

)



end_quarter_prices = df.loc[df['Date'].isin(end_quarter_dates), 'Close']



end_quarter_dates

tmp['end_quarter_date'] = end_quarter_dates.to_list()

tmp['end_quarter_price'] = end_quarter_prices.to_list()





# Calculate benchmark cumulative shares and value

tmp['benchmark_cum_shares_with_div'] = get_cum_shares_with_div(

    tmp['benchmark'], 

    tmp['end_quarter_price'], 

    dividend_yield=quarterly_dividend_yield

)

tmp['benchmark_value_at_quarter_end'] = round((tmp['benchmark_cum_shares_with_div'] * tmp['end_quarter_price']) / 100, 2)





# Calculate lower_limit cumulative shares and value

tmp['lower_limit_cum_shares_with_div'] = get_cum_shares_with_div(

    tmp['lower_limit'], 

    tmp['end_quarter_price'], 

    dividend_yield=quarterly_dividend_yield

)

tmp['lower_limit_value_at_quarter_end'] = round((tmp['lower_limit_cum_shares_with_div'] * tmp['end_quarter_price']) / 100, 2)





# Calculate upper_limit cumulative shares and value

tmp['upper_limit_cum_shares_with_div'] = get_cum_shares_with_div(

    tmp['upper_limit'], 

    tmp['end_quarter_price'], 

    dividend_yield=quarterly_dividend_yield

)

tmp['upper_limit_value_at_quarter_end'] = round((tmp['upper_limit_cum_shares_with_div'] * tmp['end_quarter_price']) / 100, 2)





tmp.head()
fig = go.Figure()





fig.add_trace(go.Scatter(

                x=tmp['end_quarter_date'],

                y=tmp['benchmark_value_at_quarter_end'],

                name="benchmark",

                line_color='grey',

                opacity=0.8)

             )





fig.add_trace(go.Scatter(

                x=tmp['end_quarter_date'],

                y=tmp['upper_limit_value_at_quarter_end'],

                name="upper_limit",

                line_color='deepskyblue',

                opacity=0.8)

             )





fig.add_trace(go.Scatter(

                x=tmp['end_quarter_date'],

                y=tmp['lower_limit_value_at_quarter_end'],

                name="lower_limit",

                line_color='red',

                opacity=0.8)

             )



# Use date string to set xaxis range

fig.update_layout(

    title_text="Total no. of Shares by year",

#     xaxis_range=['2016-07-01','2016-12-31'],

)





fig.show()
# tmp = df.copy()

# tmp['50_day_ma'] = df['Close'].rolling(50).mean()

# tmp['200_day_ma'] = df['Close'].rolling(200).mean()



# tmp = tmp[tmp['year'] >= 2010]



# tmp['price_below_50_day_ma'] = np.where(tmp['Close'] < tmp['50_day_ma'], 1, 0)



# tmp = tmp[tmp['price_below_50_day_ma'] == 1]