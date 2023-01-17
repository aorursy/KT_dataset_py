import pandas as pd
import pylab
import statsmodels.api as sm

milk = pd.read_csv('../input/monthly-milk-production.csv',';', index_col=['month'], parse_dates=['month'], dayfirst=True)
print(sm.tsa.stattools.adfuller(milk['milk']))
milk['adjmilk'] = milk.milk / milk.index.days_in_month
print (sum(milk.adjmilk))
pylab.plot(milk['adjmilk'])