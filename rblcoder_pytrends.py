!pip install pytrends

import pytrends

#https://github.com/GeneralMills/pytrends/blob/master/examples/example.py

from pytrends.request import TrendReq

pytrend = TrendReq()
from datetime import date

today = date.today()
pytrend.trending_searches(pn='india') 
pytrend.suggestions('Machine Learning')
_=pytrend.get_historical_interest(['Machine learning'], year_start=2018, month_start=1, day_start=1, hour_start=0, year_end=2018, month_end=2, day_end=1, hour_end=0, cat=0, geo='', gprop='', sleep=0)['Machine learning'].plot()

pytrend.categories()
