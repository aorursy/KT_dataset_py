import math

import numpy as np

import pandas as pd

import plotly.express as px

import plotly.figure_factory as ff

import plotly.graph_objects as go

import plotly.io as pio

import scipy.stats as stats



from plotly.subplots import make_subplots



pio.templates.default = 'plotly_white'



colors = ['#1E3C71', '#04BF95', '#FC9480', '#FFC970']

colors_binary = ['#1E3C71', '#04BF95']
url = 'http://www.econ.yale.edu/~shiller/data/ie_data.xls'

spreadsheet_df = pd.read_excel(url, sheet_name='Data', skiprows=7, usecols=['Date', 'P'])
prices_df = spreadsheet_df.dropna()

prices_df.columns = ['month', 'price']



MONTHS_IN_100_YEARS = 100 * 12

FIRST_MONTH = '1920-01-01'

LAST_START_MONTH = '2009-12-01'

prices_df = prices_df.assign(month=pd.to_datetime(prices_df['month'].apply('{:.2f}'.format), format='%Y.%m'))

prices_df = prices_df.query(f'month >= "{FIRST_MONTH}"').head(MONTHS_IN_100_YEARS)

prices_df.head()
fig = px.line(prices_df, x='month', y='price', line_shape='linear',

              labels=dict(price='Price', month=''),

              title='S&P 500 since 1920 (logarithmic scale)')

fig.update_layout(yaxis_type='log')

fig.show()
MONTHS_IN_TEN_YEARS = 10 * 12



ten_years_df = prices_df.copy()

ten_years_df = ten_years_df.assign(ten_years_price=ten_years_df.price.shift(periods=-MONTHS_IN_TEN_YEARS))

ten_years_df = ten_years_df.assign(ten_years_return=ten_years_df.ten_years_price/ten_years_df.price)

ten_years_df = ten_years_df.assign(ten_years_return_is_positive=ten_years_df.ten_years_return>1)

ten_years_completed_df = ten_years_df.dropna()

ten_years_completed_df.tail()
ten_years_results_df = ten_years_completed_df.groupby('ten_years_return_is_positive').size().to_frame('periods')

ten_years_results_df['pct'] = ten_years_results_df.periods.transform(lambda x: 100*x/sum(x)).round(1)

ten_years_results_df
ten_years_completed_df.ten_years_return.mean().round(4)
fig = px.histogram(ten_years_completed_df, x='ten_years_return', marginal='box',

  labels=dict(ten_years_return='Ten years return multiplier'), nbins=50,

  title=f'Periods by the return after holding 10 years ({ten_years_results_df.pct[True]}% of the periods are positive)')

fig.update_layout(showlegend=False)

fig.show()
fig = px.line(ten_years_completed_df, x='month', y='ten_years_return',

  labels=dict(ten_years_return='Ten years return multiplier', month='Investment start'),

  title=f'Returns after holding 10 years ({ten_years_results_df.pct[True]}% of the periods are positive)')

fig.add_trace(go.Scatter(

    x=[ten_years_completed_df.month.min(), ten_years_completed_df.month.max()],

    y=[1, 1], mode='lines'))

fig.update_layout(showlegend=False)

fig.show()
INVESTMENT_PER_MONTH = 1



dca_df = ten_years_df.copy()

dca_df = dca_df.assign(quantity_bought=INVESTMENT_PER_MONTH/dca_df.price)

dca_df = dca_df.assign(quantity_at_sell=dca_df.quantity_bought.rolling(MONTHS_IN_TEN_YEARS).sum().shift(1-MONTHS_IN_TEN_YEARS))

dca_df = dca_df.dropna()

dca_df = dca_df.assign(money_at_sell=dca_df.quantity_at_sell*dca_df.ten_years_price)

dca_df = dca_df.assign(investment_cost=INVESTMENT_PER_MONTH*MONTHS_IN_TEN_YEARS)

dca_df = dca_df.assign(dca_return=dca_df.money_at_sell/dca_df.investment_cost)

dca_df = dca_df.assign(dca_return_is_positive=dca_df.dca_return>1)

dca_df.tail()
dca_results_df = dca_df.groupby('dca_return_is_positive').size().to_frame('periods')

dca_results_df['pct'] = dca_results_df.periods.transform(lambda x: 100*x/sum(x)).round(1)

dca_results_df
dca_df.dca_return.mean().round(4)
fig = px.histogram(dca_df, x='dca_return', marginal='box', nbins=50,

  labels=dict(dca_return='Ten years return multiplier'),

  title=f'Periods by the return after holding 10 years ({dca_results_df.pct[True]}% of the periods are positive)')

fig.update_layout(showlegend=False)

fig.show()
fig = px.line(dca_df, x='month', y='dca_return',

  labels=dict(dca_return='Ten years return multiplier', month='Investment start'),

  title=f'Returns after holding 10 years ({dca_results_df.pct[True]}% of the periods are positive)')

fig.add_trace(go.Scatter(

    x=[dca_df.month.min(), dca_df.month.max()], y=[1, 1], mode='lines'))

fig.update_layout(showlegend=False)

fig.show()
dip_percentages = [0, 2, 5, 10]

dip_percentages
dip_df = prices_df.copy()

dip_df.tail()
aux_df = pd.DataFrame(columns=['start_month', 'dip', 'savings', 'high', 'curr_price'])

buys_df = pd.DataFrame(columns=['start_month', 'dip', 'buy_month', 'price', 'quantity', 'cost'])
def update_aux(aux_df, row):

  for dip in dip_percentages:

    aux_df = aux_df.append({

        'start_month': row['month'],

        'dip': dip,

        'savings': 0,

        'high': 999_999,

        'curr_price': row['price'],

    }, ignore_index=True)

  

  aux_df['savings'] += INVESTMENT_PER_MONTH

  aux_df['curr_price'] = row['price']

  aux_df['high'] = aux_df[['high', 'curr_price']].max(axis=1)



  # remove months older than 10 years

  limit_month = row['month'].replace(year=row['month'].year-10)

  aux_df = aux_df.query(f'start_month > "{limit_month}"')

  

  return aux_df





def buy(aux_df, buys_df):

  # prepare

  aux_df['price_to_buy'] = aux_df['high'] * (1 - aux_df['dip']/100)

  aux_df['should_buy'] = aux_df['curr_price'] <= aux_df['price_to_buy']



  # buy

  curr_buys_df = aux_df.query('should_buy').copy()

  curr_buys_df['buy_month'] = curr_buys_df['start_month'].max()

  curr_buys_df['price'] = curr_buys_df['curr_price']

  curr_buys_df['quantity'] = curr_buys_df['savings']/curr_buys_df['curr_price']

  curr_buys_df['cost'] = curr_buys_df['savings']

  curr_buys_df = curr_buys_df[['start_month', 'dip', 'buy_month', 'price', 'quantity', 'cost']]

  

  buys_df = pd.concat([buys_df, curr_buys_df])



  # reset

  aux_df['high'] = np.where(aux_df['should_buy'], aux_df['curr_price'], aux_df['high'])

  aux_df['savings'] = np.where(aux_df['should_buy'], 0, aux_df['savings'])



  return aux_df, buys_df
for _, row in dip_df.iterrows():

  aux_df = update_aux(aux_df, row)

  aux_df, buys_df = buy(aux_df, buys_df)



# remove periods with less than 10 years

buys_df = buys_df.query(f'start_month <= "{LAST_START_MONTH}"')
def plot_10_years(start_month, dip):

  plot_df = dip_df.query(f'month >= "{start_month}"').head(MONTHS_IN_TEN_YEARS)

  buys_plot_df = buys_df.query(f'start_month == "{start_month}" and dip == {dip}')



  fig = px.line(plot_df, x='month', y='price', line_shape='linear',

                title=f'Example of 10 years of investment using {dip}% dip starting in {start_month}',

                labels=dict(price='Price', month=''))

  fig.add_trace(go.Scatter(

          x=buys_plot_df.buy_month,

          y=buys_plot_df.price,

          mode='markers',

          name='Buys',

      ))

  fig.show()
start_month = '2001-05-01'

dip = 5

plot_10_years(start_month, dip)
sums_df = buys_df.groupby(['start_month', 'dip'])[['quantity', 'cost']].sum()

total_buys_df = buys_df.groupby(['start_month', 'dip']).size().to_frame('buys')

dip_results_df = pd.merge(sums_df, total_buys_df, left_index=True, right_index=True).reset_index()

dip_results_df = dip_results_df.merge(

    ten_years_df[['month', 'ten_years_price']],

    left_on='start_month', right_on='month')

dip_results_df.head()
fig = px.histogram(dip_results_df, x='buys', marginal='box',

                   color='dip', nbins=50, barmode='group',

                   labels=dict(buys='Number of buys'),

                   title='Buys performed during 10 years',

                   color_discrete_sequence=colors)

fig.show()
dip_results_df = dip_results_df.assign(investment_cost=INVESTMENT_PER_MONTH*MONTHS_IN_TEN_YEARS)

money_at_sell = (

    dip_results_df.quantity*dip_results_df.ten_years_price

    + dip_results_df.investment_cost - dip_results_df.cost

) 

dip_results_df = dip_results_df.assign(money_at_sell=money_at_sell)

dip_results_df = dip_results_df.assign(dip_return=dip_results_df.money_at_sell/dip_results_df.investment_cost)

dip_results_df = dip_results_df.assign(dip_return_is_positive=dip_results_df.dip_return>1)

dip_results_df.head()
fig = px.histogram(dip_results_df, x='dip_return', marginal='box',

                   color='dip', nbins=50, barmode='group',

                   labels=dict(dip_return='Ten years return multiplier'),

                   title='Periods by the return after holding 10 years',

                   range_x=[dip_results_df.dip_return.min()-0.1, dip_results_df.dip_return.max()+0.1],

                   color_discrete_sequence=colors)

fig.show()
fig = px.line(dip_results_df, x='start_month', y='dip_return', color='dip',

              title='Returns of the different strategies',

              color_discrete_sequence=colors,

              labels=dict(dip_return='Return multiplier', start_month='Date of investment start'))

fig.show()
postivite_return_pct_df = pd.DataFrame(columns=['dip', 'positive_pct'])



for dip in dip_results_df.dip.unique():

  pct = dip_results_df.query(f'dip == {dip}').groupby('dip_return_is_positive').size().transform(lambda x: 100*x/sum(x))

  postivite_return_pct_df = postivite_return_pct_df.append({

      'dip': f'{dip}%',

      'positive_pct': f'{pct[True].round(2)}%',

  }, ignore_index=True)



postivite_return_pct_df
dip_returns_df = (

    dip_results_df[['start_month', 'dip', 'dip_return']]

    .pivot(index='start_month', columns='dip', values='dip_return')

)



for dip in dip_percentages:

  if dip == 0:

    continue



  dip_returns_df[f'dip_{dip}_performance'] = dip_returns_df[dip] / dip_returns_df[0]



heatmap_comparative_df = dip_returns_df[[0, 2, 5, 10]]

heatmap_comparative_df[2] = heatmap_comparative_df[2] > heatmap_comparative_df[0]

heatmap_comparative_df[5] = heatmap_comparative_df[5] > heatmap_comparative_df[0]

heatmap_comparative_df[10] = heatmap_comparative_df[10] > heatmap_comparative_df[0]

heatmap_comparative_df = heatmap_comparative_df[[2, 5, 10]].astype(int)

heatmap_comparative_df.tail()
months = np.flip(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

years = list(range(1920, 2010))



dip_values = heatmap_comparative_df.columns



subplot_titles = []

for dip in dip_values:

  outperforming_months = heatmap_comparative_df[dip].sum()

  subplot_titles.append(f'Dip {dip}% beated DCA in {outperforming_months} months')



fig = make_subplots(rows=len(dip_values), cols=1,

                    subplot_titles=subplot_titles)



for idx, dip in enumerate(dip_values):

  values = np.reshape(heatmap_comparative_df[dip].to_numpy(), (12, 90), order='F')

  values = np.flipud(values)



  data=go.Heatmap(

      name=f'Dip {dip}%',

      x=years,

      y=months,

      z=values,

      colorscale=colors_binary,

      showscale=False

  )



  fig.append_trace(data, row=idx+1, col=1)



fig.update_layout(title='Months Buying the Dip outperformed DCA')

fig.show()
p = 0.5

n_bets = len(heatmap_comparative_df)

binomial = stats.binom(n_bets, p)

binomial_df = pd.DataFrame(columns=['dip', 'p_value'])



for dip in heatmap_comparative_df.columns:

  n_successful_bets = heatmap_comparative_df[dip].sum()



  p_value = 0

  for k in range(1, n_successful_bets + 1):

      p_value += binomial.pmf(k)



  binomial_df = binomial_df.append({

      'dip': dip,

      'p_value': p_value,

  }, ignore_index=True)



binomial_df
metrics_df = pd.DataFrame(columns=['metric', 'mean', 'std', 'avg_lift', 'p_value'])

lifts = {}



for dip in [2, 5, 10]:

  n_simulations = 2_000

  n_samples = 1_000



  lifts[dip] = []



  for _ in range(n_simulations):

    dca_samples = dip_returns_df[0].sample(n_samples, replace=True)

    dip_samples = dip_returns_df[dip].sample(n_samples, replace=True)

    lifts[dip].append(dip_samples.mean()/dca_samples.mean())

  

  metric = f'dip_{dip}_performance'

  values = lifts[dip]

  p_value = np.mean(np.array(values)>1)

  metrics_df = metrics_df.append({

      'metric': metric,

      'mean': np.mean(dip_returns_df[dip]).round(6),

      'std': np.std(dip_returns_df[dip]).round(6),

      'avg_lift': round(100 * (np.mean(values) - 1), 2),

      'p_value': min(p_value, 1-p_value),

  }, ignore_index=True)



metrics_df.round(6)
metrics_df = pd.DataFrame(columns=['metric', 'mean', 'std', 'avg_lift', 'p_value'])

means = {}

means_95 = {}



for dip in [0, 2, 5, 10]:

  n_simulations = 2_000

  n_samples = 1_000



  dip_means = []



  for _ in range(n_simulations):

    # Sample

    dip_samples = dip_returns_df[dip].sample(n_samples, replace=True)

    dip_means.append(dip_samples.mean())



  # Get the original means

  means[dip] = dip_means



  # Remove the 95% of the means

  dip_means = np.array(dip_means)

  upper_limit = np.percentile(dip_means, 97.5)

  lower_limit = np.percentile(dip_means, 2.5)

  indexes = np.where(np.logical_and(dip_means>=lower_limit, dip_means<=upper_limit))[0]

  means_95[dip] = dip_means[indexes]
hist_data = list(means.values())

group_labels = list(map(lambda x: f'{x}% Dip' if x != 0 else 'DCA', means.keys()))



# Create distplot with custom bin_size

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)

fig.update_layout(title='Means of the strategy returns for each simulation')

fig.show()
hist_data = list(means_95.values())

group_labels = list(map(lambda x: f'{x}% Dip' if x != 0 else 'DCA', means_95.keys()))



# Create distplot with custom bin_size

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)

fig.update_layout(title='Only taking 95% of the means')

fig.show()