import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
# !pip uninstall pytrends --yes

!pip install pytrends
from pytrends.request import TrendReq

pytrend = TrendReq()
# An example getting historical/real-time search interest from Google Trends.

kw_list=['insomnia', 'anxiety', 'depression']

pytrend.get_historical_interest(kw_list, year_start=2020, month_start=5, day_start=1, hour_start=1, year_end=2020, month_end=6, day_end=20, hour_end=23, cat=0, geo='', gprop='', sleep=0) 
# Line chart plot

kw_list = ['counselling', 'psychiatrist']

interest = pytrend.get_historical_interest(kw_list, year_start=2020, month_start=2, day_start=1, hour_start=1, year_end=2020, month_end=3, day_end=31, hour_end=23, cat=0, geo='', gprop='', sleep=0) 

plt.figure(figsize=(20, 7))

plt.plot(interest)



plt.figure(figsize=(20, 7))

kw_list = ['ocd', 'depression', 'anxiety']

interest = pytrend.get_historical_interest(kw_list, year_start=2020, month_start=4, day_start=1, hour_start=1, year_end=2020, month_end=5, day_end=31, hour_end=23, cat=0, geo='', gprop='', sleep=0) 

plt.plot(interest)
# Interest by region

pytrend.build_payload(kw_list=['ocd', 'anxiety', 'depression', 'panic attack', 'mental health'])

df = pytrend.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False)

df.head(20)
# Related queires

pytrend.build_payload(kw_list=['ocd', 'anxiety', 'depression', 'panic attack', 'mental health'])

related_queries = pytrend.related_queries()

related_queries.values()
pytrend.build_payload(kw_list=['mental health coronavirus'])

related_queries = pytrend.related_queries()

related_queries.values()
pytrend.build_payload(kw_list=['anxiety coronavirus'])

related_queries = pytrend.related_queries()

related_queries.values()
df.reset_index().plot(x='geoName', y='mental health', figsize=(50, 10), kind='bar')