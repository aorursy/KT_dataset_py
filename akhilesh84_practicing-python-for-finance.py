import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import mpl_finance as mpf

%matplotlib inline
def PV(FV, r, T):

    """Returns present value given a future value"""

    return FV / ((1 + r) ** T)



def FV(PV, r, T):

    """Returns future value given a present value. This is basically a compound interest calculation implementation"""

    return PV * ((1 + r)**T)



def compound_interest_with_annual_premium(P, r, T):

    yearly_investment = np.ones(T) * P

    return sum([P * ((1 + r) ** t) for (P,t) in zip(yearly_investment, range(T, 0, -1))])



def amortization_sheet(loan_amount, rate_of_interest, term, extra_payment = 0):

    sheet = pd.DataFrame({

        'Balance': np.zeros(term * 12),

        'Principal': np.zeros(term * 12),

        'Interest': np.zeros(term * 12),

        'Cum_Interest': np.zeros(term * 12),

        'Outstanding': np.zeros(term * 12)

    })

    

    pmt = np.pmt(rate_of_interest/12, term * 12, -loan_amount, fv=0)

    

    print(pmt)

    

    sheet['Balance'][0] = loan_amount

    sheet['Interest'][0] = loan_amount * (rate_of_interest/12)

    sheet['Principal'][0] = pmt - sheet['Interest'][0]

    sheet['Outstanding'][0] = loan_amount - sheet['Principal'][0]

    sheet['Cum_Interest'] = sheet['Interest'].cumsum()

    

    for payment in range(1, term * 12 - 1):

        sheet['Balance'][payment] = sheet['Outstanding'][payment-1]

        sheet['Interest'][payment] = sheet['Balance'][payment] * (rate_of_interest/12)

        sheet['Principal'][payment] = pmt - sheet['Interest'][payment]

        sheet['Outstanding'][payment] = sheet['Balance'][payment] - sheet['Principal'][payment]

    sheet['Cum_Interest'] = sheet['Interest'].cumsum()

    

    return round(sheet, 2)
# amortization_sheet(4500000, 0.0875, 20).head()

amort_sheet = amortization_sheet(4500000, 0.0850, 20)

amort_sheet
plt.plot(amort_sheet['Principal'][:-1])

plt.plot(amort_sheet['Interest'])

plt.legend()
amortization_sheet(4500000, 0.0875, 20).tail()
ms = pd.read_csv('../input/MS.csv', index_col='Date', parse_dates=['Date'])

ms.drop(columns=['Adj Close'], axis=1, inplace=True)

ms = ms.resample('W').last()['2009':]

# ms.info()
ms.head()
ax = plt.axes()

plt.title('Open-High-Low-Close Chart')

ax.set_xlabel('Date')

ax.set_ylabel('Closing Price')

data = ms.loc['2009':]

mpf.candlestick2_ohlc(ax = ax, opens=data['Open'], highs=data['High'], lows=data['Low'], closes=data['Close']);
ms['Change'] = np.log(ms.Close / ms.Close.shift(periods=1))

ms.loc['2009':]['Change'].plot(kind='kde');
np.std(ms.loc['2009':]['Change']), np.mean(ms.loc['2009':]['Change'])
# A simple simulation where we are considering a weekly change derieved from a noral distribution to predict the values in future.

# In practice, a large number of simulations are run. Each run is represented by an iteration



change = np.random.normal(loc=0.001, scale=0.05, size=520)



P = 42.990000

final_val = np.zeros(520)



for i in range(len(change)):

    P = P * (1 + change[i])

    final_val[i] = P