import numpy as np

import pandas as pd

import datetime



import seaborn as sns

sns.set(style="white", color_codes=True)



import matplotlib.pyplot as plt

import matplotlib.lines as mlines

import sqlite3







d = pd.read_csv("../input/mass_shootings_all.csv")


g=d[['State','# Killed', '# Injured']].groupby(['State']).agg(['sum','count'])
# Reference: 

#  http://www.census.gov/popest/data/state/totals/2015/tables/NST-EST2015-01.csv

population = {"Alabama":4858979,

"Alaska":738432,

"Arizona":6828065,

"Arkansas":2978204,

"California":39144818,

"Colorado":5456574,

"Connecticut":3590886,

"Delaware":945934,

"District of Columbia":672228,

"Florida":20271272,

"Georgia":10214860,

"Hawaii":1431603,

"Idaho":1654930,

"Illinois":12859995,

"Indiana":6619680,

"Iowa":3123899,

"Kansas":2911641,

"Kentucky":4425092,

"Louisiana":4670724,

"Maine":1329328,

"Maryland":6006401,

"Massachusetts":6794422,

"Michigan":9922576,

"Minnesota":5489594,

"Mississippi":2992333,

"Missouri":6083672,

"Montana":1032949,

"Nebraska":1896190,

"Nevada":2890845,

"New Hampshire":1330608,

"New Jersey":8958013,

"New Mexico":2085109,

"New York":19795791,

"North Carolina":10042802,

"North Dakota":756927,

"Ohio":11613423,

"Oklahoma":3911338,

"Oregon":4028977,

"Pennsylvania":12802503,

"Rhode Island":1056298,

"South Carolina":4896146,

"South Dakota":858469,

"Tennessee":6600299,

"Texas":27469114,

"Utah":2995919,

"Vermont":626042,

"Virginia":8382993,

"Washington":7170351,

"West Virginia":1844128,

"Wisconsin":5771337,

"Wyoming":586107,}
g.head()
g.columns = ["k_sum","k_count","i_sum","i_count"]

g['Population'] = g.index  # Use State for getting population of State

g['Population']=g['Population'].apply(lambda x: population[x])

g['pp']=g['k_count']/g['Population']
# Mass shootings by state order by highest counts per population

g.sort_values(by='pp',ascending=False,inplace=True)

g