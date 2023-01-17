import pandas as pd
import numpy as np
import matplotlib.pyplot
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/all_stocks_5yr.csv',parse_dates = ['date'], index_col=False)
close = df.reset_index().pivot(index = 'date',columns = 'Name',values='close')
#close
# resampling data to monthly frequency
freq = 'M'
monthly_close = close.resample(freq).last()
#monthly_close
#monthly_close.plot(x=index, y='AAL')
#Getting the returns
monthly_log_returns = np.log(monthly_close)-np.log(monthly_close.shift(1))
prev_returns = monthly_log_returns.shift(1)
lookahead_returns = monthly_log_returns.shift(-1)
# getting long and short portfolios
def get_top_n(prev_returns, top_n):

    p_r = prev_returns.copy()
    for i, row in p_r.iterrows():
        top = row.nlargest(top_n).index
        p_r.loc[i] = 0
        p_r.loc[i,top] = 1
    print(p_r.astype('int64'))
    return p_r.astype('int64')
top_bottom_n = 50
df_long = get_top_n(prev_returns, top_bottom_n)
df_short = get_top_n(-1*prev_returns, top_bottom_n)
# We'll assume every stock gets an equal dollar amount of investment.
expected_portfolio_returns = lookahead_returns*(df_long-df_short)/(2*top_bottom_n)
expected_portfolio_returns_by_date = expected_portfolio_returns.T.sum().dropna()
portfolio_ret_mean = expected_portfolio_returns_by_date.mean()
portfolio_ret_ste = expected_portfolio_returns_by_date.sem()
portfolio_ret_annual_rate = (np.exp(portfolio_ret_mean * 12) - 1) * 100

print("""
Mean:                       {:.6f}
Standard Error:             {:.6f}
Annualized Rate of Return:  {:.2f}%
""".format(portfolio_ret_mean, portfolio_ret_ste, portfolio_ret_annual_rate))

# Statistical Analysis
from scipy import stats
t, p_two_tail = stats.ttest_1samp(expected_portfolio_returns_by_date,0)
print("""
Alpha analysis:
 t-value:        {:.3f}
 p-value:        {:.6f}
""".format(t, p_two_tail/2))
