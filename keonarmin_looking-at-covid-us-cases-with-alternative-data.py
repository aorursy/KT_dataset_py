#installing google trends unofficial API

!pip install pytrends



import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure

import numpy as np

from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US', tz=360)



kw_list = ["covid symptoms"]
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-CA', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('California')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-13', geo='US-NY', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('New York')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-TN', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Tennesse')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-AL', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Alabama')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-FL', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Florida')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-OR', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Oregon')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-AK', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Alaska')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-AZ', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Arizona')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-AR', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Arkansas')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-CO', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Colorado')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-CT', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Connecticut')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-DE', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Delaware')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-DC', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('District of Columbia')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-GA', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Georgia')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-HI', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Hawaii')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-ID', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Idaho')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-IL', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Illinois')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-IN', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Indiana')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-IA', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Iowa')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-KS', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Kansas')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-KY', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Kentucky')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-LA', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Louisiana')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-ME', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Maine')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-MD', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Maryland')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-LA', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Louisiana')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-MA', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Massachusetts')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-MI', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Michigan')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-MN', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Minnesota')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-MS', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Mississippi')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-MO', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Missouri')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-MT', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Montana')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-NE', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Nebraska')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-NH', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('New Hampshire')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-NV', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Nevada')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-NJ', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('New Jersey')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-NM', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('New Mexico')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-NC', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('North Carolina')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-ND', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('North Dakota')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-OH', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Ohio')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-OK', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Oklahoma')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-RI', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Rhode Island')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-PA', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Pennsylvania')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-SC', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('South Carolina')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-SD', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('South Dakota')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-TX', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Texas')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-UT', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Utah')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-VT', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Vermont')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-VA', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Virginia')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-WA', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Washington')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-WV', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('West Virginia')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-WI', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Wisconsin')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )
payload = pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-01 2020-06-12', geo='US-WY', gprop='')

interest = pytrends.interest_over_time()

interest.plot()

plt.title('Wyoming')

print( interest['covid symptoms'][102] - interest['covid symptoms'][91] / interest['covid symptoms'][102] )