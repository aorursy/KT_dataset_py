#!pip install pytrends

import pandas as pd
from pytrends.request import TrendReq
pytrend = TrendReq(hl='en-US', tz=360)
# Create payload and capture API tokens. Only needed for interest_over_time(), interest_by_region() & related_queries()trendy_keywords = ‘Data Science’

trendy_keywords = 'Data Science'
kw_list = [trendy_keywords]
kw = trendy_keywords
pytrend.build_payload(kw_list)
pytrend.build_payload(kw_list)
pytrend.interest_over_time()
pytrend.interest_by_region()
# Related Queries, returns a dictionary of dataframes
related_queries_dict = pytrend.related_queries()
print(related_queries_dict)
# for rising related queries
related_queries_rising = related_queries_dict.get(kw).get('rising')

# for top related queries
related_queries_top = related_queries_dict.get(kw).get('top')
print("**************** RISING RELATED KEYWORDS **************")
print(related_queries_rising)
print("**************** TOP RELATED KEYWORDS *******************")
print(related_queries_top)