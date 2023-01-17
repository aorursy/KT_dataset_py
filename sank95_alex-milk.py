import pandas as pd
import pylab
import calendar
import statsmodels.api as sm
def dni(milk):
    new_list = []
    for m in milk.index:
        new_list.append(calendar.monthrange(m.year, m.month)[1])
    return  new_list

milk = pd.read_csv('../input/monthly-milk-production.csv',';', index_col=['month'], parse_dates=['month'], dayfirst=True)
print(sm.tsa.stattools.adfuller(milk['milk']))
day = dni(milk)
milk['milk'] = list(map(lambda x,y: x / y, milk['milk'], day))
pylab.plot(milk)
print(sum(milk['milk']))