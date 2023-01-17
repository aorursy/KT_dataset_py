%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
df_crsp = pd.read_csv('../input/week-2/monthly.csv')  # Load part of the file to have a look
df_crsp.info()
df_crsp.head()
df_crsp = pd.read_csv('../input/week-2/monthly.csv', na_values=['C'], parse_dates=['DATE'])

# The output from WRDS returns a mixed of small and large cap column names. Let's standardize everything in small caps.
cols = df_crsp.columns
print(cols)
print(type(cols))

# List comprehension [expression for variable in iterable]
df_crsp.columns = [c.lower() for c in cols]
df_crsp.info()
df_crsp.head()
df_crsp.describe()
# Sanitize dataset

# Drop observations with missing returns
df_crsp = df_crsp[df_crsp.ret.notnull()]

# Take the absolute value of the price
df_crsp['prc'] = np.abs(df_crsp['prc'])
# Set the index (to select easily on date)
df_crsp = df_crsp.set_index('date')
df_crsp = df_crsp.sort_index()
# Compute continuously compounded returns (i.e. log returns).
df_crsp['lret'] = np.log(1 + df_crsp['ret'])
df_crsp['lvwretd'] = np.log(1 + df_crsp['vwretd'])
df_crsp['lewretd'] = np.log(1 + df_crsp['ewretd'])

# Compute the market cap
df_crsp['size'] = df_crsp['shrout'] * df_crsp['prc']
df_crsp.head()

# Total market size over time

ax = df_crsp.groupby(['date'])['size'].sum().plot()
# Taking the log of market size
ax = df_crsp.groupby(['date'])['size'].sum().plot(logy=True)

# Add some informative lines
ax.axvline(x=datetime(1929,10,24), color='k', linestyle=':') # Black Monday
ax.axvline(x=datetime(1987,10,19), color='k', linestyle=':') # Black Monday again
# Parameters
form_period = 36 # 36 Formation period, in month
start_date = '1933-01-01' 
end_date = '2000-01-01' 
# Get the dates of portfolio formation.
# The frequency tells how far apart to put the dates.
# 'M' stand for month, 'MS' is for month start, to make sure we 
# have first day of the month. It needs to be a string, so we convert
# our numbers to string. 
dates = pd.date_range(start=start_date, end=end_date, freq=str(np.int(form_period)) + 'M')
dates
form_period = 36 # 36 Formation period, in month
n_stocks = 35  # Number of stocks in the top and bottom performance
benchmark = 'vwretd' # Benchmark market return to use ('vwretd' or 'ewretd')
# Let's first do it for only one date.
date = dates[0]
print(dates[0])

beg_dt = date - pd.offsets.MonthEnd(1) * form_period

# Select obs for the formation period
crsp_t = df_crsp[beg_dt:date].copy()
crsp_t
# We only want to keep stocks that are there during the full formation window

crsp_t['N'] = crsp_t.groupby(['permno'])['permno'].transform('count')

# Filter on number of observations. We only keep sotcks for which we have returns
# over the full observation period.
crsp_t = crsp_t[crsp_t['N'] >= form_period]

# Now for each stock we want to compute the full period return.
stock_ret = crsp_t.groupby('permno')[['lret', 'lvwretd', 'lewretd']].sum()
stock_ret.head()
# Next compute excess returns based on the chosen index.
# Note that since the benchmark is the same for all stocks, we could use
# actual returns for ranking purposes. It would only make a difference in some
### cases. Which ones?

stock_ret['lexret'] = stock_ret['lret'] - stock_ret['l' + benchmark]
# Now rankings.

stock_ret['rank_asc'] = stock_ret['lexret'].rank() # (1 = worst return)
stock_ret['rank_inv'] = stock_ret['lexret'].rank(ascending=False) # (1= best return)
stock_ret
# Assign stock to top or bottom portfolio

top_portfolio = stock_ret[stock_ret.rank_inv <= n_stocks].reset_index()[['permno', 'lexret']]
bottom_portfolio = stock_ret[stock_ret.rank_asc <= n_stocks].reset_index()[['permno', 'lexret']]
    
top_portfolio.head()

bottom_portfolio.head()
def compute_performance_portfolios(date, df, form_period=36, n_stocks=35,
                                   benchmark='vwretd'):
    beg_dt = date - pd.offsets.MonthBegin(1) * form_period

    # Select obs for the formation period
    crsp_t = df[beg_dt:date].copy()
    
    # We only want to keep stocks that are there during the full formation window
    crsp_t['N'] = crsp_t.groupby(['permno'])['permno'].transform('count')
    # Filter on number of observations. We only keep sotcks for which we have returns
    # over the full observation period.
    crsp_t = crsp_t[crsp_t['N'] >= form_period]
    
    # Now for each stock we want to compute the full period return. Easy with log returns, just sum up!
    stock_ret = crsp_t.groupby('permno')[['lret', 'lvwretd', 'lewretd']].sum()
    
    # Next compute excess returns based on the chosen index.
    # Note that since the benchmark is the same for all stocks, we could use
    # actual returns for ranking purposes.
    stock_ret['lexret'] = stock_ret['lret'] - stock_ret['l' + benchmark]
    
    # Now rankings.
    stock_ret['rank_asc'] = stock_ret['lexret'].rank() # (1 = worst return)
    stock_ret['rank_inv'] = stock_ret['lexret'].rank(ascending=False) # (1= best return)
    
    # Assign stock to top or bottom portfolio
    top_portfolio = stock_ret[stock_ret.rank_inv <= n_stocks].reset_index()[['permno', 'lexret']]
    bottom_portfolio = stock_ret[stock_ret.rank_asc <= n_stocks].reset_index()[['permno', 'lexret']]
    
    return (bottom_portfolio, top_portfolio)
portfolios = {}
for date in dates:
    portfolios[date] = compute_performance_portfolios(date, df_crsp, benchmark='vwretd')
type(portfolios)
portfolios.keys()
portfolios.values()
portfolios[date][0].head() #loser portfolio from  "return (bottom_portfolio, top_portfolio)""
portfolios[date][1].head()
date = dates[0]
portfolio = portfolios[date][0] # Bottom portfolio.

hold_period = 36 # Holding period, in months
benchmark = 'vwretd' # 'vwretd' or 'ewretd'
weighting = 'vw' # 'vw' or 'ew'

portfolio = portfolio.copy()
end_dt = date + pd.offsets.MonthBegin(1) * hold_period

# Select obs for the formation period
crsp_t2 = df_crsp[date:end_dt].copy()
portfolio.info()
crsp_t2 #contains all stock in date range
# Merge with stocks in portfolios, to keep only those stocks
crsp_t2 = pd.merge(crsp_t2.reset_index(), portfolio, on=['permno'])
crsp_t2
crsp_t2.head()
crsp_t2.tail()
# We want to make sure we have one observation for each stock/date.
# If a stock is delisted, its returns will be 0 after it disappears,
# so we just fill in these missing values.

# The idea here is to create a DataFrame with all the permno/date pairs
# that we want in the final dataset. Then we merge that list with the
# dataset using "outer" which will generate missing values for the
# pairs that are not in the dataset.

# Get the dates in the dataset.
pairs_t2 = [{'date': d, 'permno': p} for d in crsp_t2['date'].unique() 
                                    for p in portfolio['permno'].unique()]
print(pairs_t2[:10])
print(type(pairs_t2))
print(type(pairs_t2[0]))
print(len(portfolio))
pairs_t2 = pd.DataFrame(pairs_t2)
pairs_t2.info()
pairs_t2
# Merge to generate placeholders

crsp_t2 = pd.merge(crsp_t2, pairs_t2, how='outer', on=['permno', 'date'])
crsp_t2.info()
# Fill missing values with 0
ret_cols = ['ret', 'vwretd', 'ewretd', 'lvwretd','lewretd', 'lret', 'lexret']
crsp_t2[ret_cols] = crsp_t2[ret_cols].fillna(0.0)
# Now we want the return up to each point in time
crsp_t2['lcumret'] = crsp_t2.groupby('permno')['lret'].cumsum()
crsp_t2['lcum' + benchmark] = crsp_t2.groupby('permno')['l' + benchmark].cumsum() # that's l (letter) not 1 (number)
# At each point in time, the return of the portfolio will be the 
# cumulative return of each component weighted by the initial weight.
# Note that here we need the average (equally- or value-weighted) cumulative return
# So, we need to back out the cumulative return from its log form.

crsp_t2['cumret'] = np.exp(crsp_t2['lcumret']) - 1 #remove the log by exp lcumret
crsp_t2['cum' + benchmark] = np.exp(crsp_t2['lcum' + benchmark]) - 1

# Add weights, equal weighted is easy.
portfolio['ew'] = 1 / len(portfolio)

# For value-weighted, need to get size as of formation date.
portfolio['date'] = date
weights = pd.merge_asof(portfolio, df_crsp[['permno', 'size']],
                        by='permno',
                        left_on='date',
                        right_index=True)
weights['vw'] = weights['size'] / weights['size'].sum()

del weights['lexret']
del weights['date']
del weights['size']
weights.head()
# Now merge back with returns
crsp_t2 = pd.merge(crsp_t2, weights, on='permno')
crsp_t2.head()
# Now compute the weighted cumulative return: equally- or value-weighted
crsp_t2['wcumret'] = crsp_t2[weighting] * crsp_t2['cumret'] # thats portfolio return
crsp_t2['wcum' + benchmark] = crsp_t2[weighting] * crsp_t2['cum' + benchmark]

portfolio_ret = crsp_t2.groupby(['date'])[['wcumret', 'wcum' + benchmark]].sum() #i do not get this line
# Count the months
portfolio_ret = portfolio_ret.reset_index()
portfolio_ret['months'] = portfolio_ret.index.values + 1 # because python starts from 0
portfolio_ret['exret'] = portfolio_ret['wcumret'] - portfolio_ret['wcum' + benchmark]
portfolio_ret.head()
def compute_holding_returns(date, portfolio, df, benchmark='vwretd', weighting='vw', hold_per=36):
    portfolio = portfolio.copy()
    end_dt = date + pd.offsets.MonthBegin(1) * hold_period
    # Select obs for the formation period
    crsp_t2 = df[date:end_dt].copy()
    # Merge with stocks in portfolios, to keep only those stocks
    crsp_t2 = pd.merge(crsp_t2.reset_index(), portfolio, on=['permno'])
    crsp_t2
    

    # Get the dates in the dataset.
    pairs_t2 = [{'date': d, 'permno': p} for d in crsp_t2['date'].unique() for p in portfolio['permno'].unique()]
    pairs_t2 = pd.DataFrame(pairs_t2)
    pairs_t2
    crsp_t2 = pd.merge(crsp_t2, pairs_t2, how='outer', on=['permno', 'date'])
    ret_cols = ['ret', 'vwretd', 'ewretd', 'lvwretd','lewretd', 'lret', 'lexret']
    crsp_t2[ret_cols] = crsp_t2[ret_cols].fillna(0.0)
    
    # Now we want the return up to each point in time
    crsp_t2['lcumret'] = crsp_t2.groupby('permno')['lret'].cumsum()
    crsp_t2['lcum' + benchmark] = crsp_t2.groupby('permno')['l' + benchmark].cumsum()

    # At each point in time, the return of the portfolio will be the 
    # cumulative return of each component weighted by the initial weight.
    # Note that here we need the simple return average, not log return.
    crsp_t2['cumret'] = np.exp(crsp_t2['lcumret']) - 1
    crsp_t2['cum' + benchmark] = np.exp(crsp_t2['lcum' + benchmark]) - 1

    # Add weights, equal weighted is easy.
    portfolio['ew'] = 1 / len(portfolio)

    # For value-weighted, need to get size as of formation date.
    portfolio['date'] = date
    weights = pd.merge_asof(portfolio, df_crsp[['permno', 'size']],
                            by='permno',
                            left_on='date',
                            right_index=True)
    weights['vw'] = weights['size'] / weights['size'].sum()

    del weights['lexret']
    del weights['date']
    del weights['size']
    
    # Now merge back with returns
    crsp_t2 = pd.merge(crsp_t2, weights, on='permno')
    
    # Now compute the weighted cumulative return
    crsp_t2['wcumret'] = crsp_t2[weighting] * crsp_t2['cumret']
    crsp_t2['wcum' + benchmark] = crsp_t2[weighting] * crsp_t2['cum' + benchmark]

    portfolio_ret = crsp_t2.groupby(['date'])[['wcumret', 'wcum' + benchmark]].sum()
    
    # Count the months
    portfolio_ret = portfolio_ret.reset_index()
    portfolio_ret['months'] = portfolio_ret.index.values + 1
    
    portfolio_ret['exret'] = portfolio_ret['wcumret'] - portfolio_ret['wcum' + benchmark]
    
    return portfolio_ret
bottom_portfolio_ret = []
top_portfolio_ret = []

for date in dates:
    bottom_portfolio_ret.append(compute_holding_returns(date, portfolios[date][0], df_crsp, benchmark='vwretd', weighting='vw'))
    top_portfolio_ret.append(compute_holding_returns(date, portfolios[date][1], df_crsp, benchmark='vwretd', weighting='vw'))

bottom_portfolio_ret
type(bottom_portfolio_ret)
bottom_portfolio_ret = pd.concat(bottom_portfolio_ret)
top_portfolio_ret = pd.concat(top_portfolio_ret)
bottom_portfolio_ret
#  type(bottom_portfolio_ret)
ax = bottom_portfolio_ret.groupby('months')['exret'].mean().plot(label='Past losers')
top_portfolio_ret.groupby('months')['exret'].mean().plot(ax=ax, label='Past winners')
ax.legend()

ax.axhline(y=0,  color='black', alpha=0.5, linestyle=':')
ax.axvline(x=12, color='black', alpha=0.5, linestyle='-')
ax.axvline(x=24, color='black', alpha=0.5, linestyle='-')
# Only pre 1980
ax = bottom_portfolio_ret.set_index('date')[:'1980'].groupby('months')['exret'].mean().plot(label='Past losers')
top_portfolio_ret.set_index('date')[:'1980'].groupby('months')['exret'].mean().plot(ax=ax, label='Past winners')
ax.legend()
ax.axhline(y=0,  color='black', alpha=0.5, linestyle=':')
ax.axvline(x=12, color='black', alpha=0.5, linestyle='-')
ax.axvline(x=24, color='black', alpha=0.5, linestyle='-')
top_portfolio_ret.set_index('date')[:'1980']
tset, pval = stats.ttest_1samp(top_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'],0)

print('Mean Excess Return for Top Performing Funds ' + str(top_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'].mean()))
print('P-values ' + str(pval))
if pval < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")
tset, pval = stats.ttest_1samp(bottom_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'],0)

print('Mean Excess Return for Bottom Performing Funds ' + str(bottom_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'].mean()))
print('P-values ' + str(pval))
if pval < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")
tset, pval = stats.ttest_ind(top_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'],
                             bottom_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'])

print('Mean Excess Return for Top Performing Funds ' + str(top_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'].mean()))
print('Mean Excess Return for Bottom Performing Funds ' + str(bottom_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'].mean()))
print('Mean Difference Bottom Minus Top in Excess Return ' + 
      str(bottom_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'].mean() -                                                        
          top_portfolio_ret[top_portfolio_ret['months']==36].set_index('date')[:'1980']['exret'].mean()))
print('P-values ' + str(pval))
if pval < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")