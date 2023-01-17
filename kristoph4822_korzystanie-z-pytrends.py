import sys

#zainstalowanie pytrends dla tego notebooka

!{sys.executable} -m pip install pytrends 



from pytrends.request import TrendReq

pytrends = TrendReq(hl = 'PL')
kw_list = ["Koronawirus", "Wybory"]
kw_list = ["/m/02bft"] # kod dla tematu "Zaburzenia depresyjne"
cat = 16 # Kategoria: Wiadomości
geo = 'PL'
last_4_hours = "now 4-H"

last_day = "now 1-d"

last_3_months = "today 3-m"

last_5_years = 'today 5-y'

specific_date = "2020-03-14 2020-03-25"

specific_date_with_hours = "2020-04-06T01 2017-04-12T22"

full_timeframe = "all"
gprop = 'news'
pytrends.build_payload(kw_list, cat=cat, timeframe=last_5_years, geo=geo)
pytrends.interest_over_time()
pytrends.get_historical_interest(kw_list, year_start=2020, month_start=4, day_start=1, hour_start=0, year_end=2020, month_end=5, day_end=1, hour_end=0, cat=0, geo='PL', gprop='', sleep=0)
pytrends.interest_by_region(inc_low_vol=True, inc_geo_code=True)
pytrends.trending_searches(pn='poland') 
date = 2019

pytrends.top_charts(date, geo='PL')