import numpy as np 
import pandas as pd 
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib as mp
import bq_helper
import os
print(os.listdir("../input"))   

df = pd.read_csv("../input/astronauts.csv")
df

# df[df['Death Mission'].notnull()].shape
swc = df["Space Walks"]
data = [go.Histogram(x=swc)]
data

#py.iplot(data, filename='basic histogram')
import matplotlib 
matplotlib.pyplot.hist(swc)
matplotlib.pyplot.hist(swc[swc > 0])
matplotlib.pyplot.hist(swc[swc > 0])
year = df['Year']
yearn = np.isnan(year) == False
#yearn
matplotlib.pyplot.hist(x = year[yearn])
year = df['Year']
yearn = np.isnan(year) == False
#yearn
matplotlib.pyplot.hist(x = year[yearn], bins = range(1955,2010))
year.sort_values()
allyears = year.sort_values()
year70s = allyears.iloc[range(60,100)]
year70s
firstnames = df['Name'].str.split().apply(lambda x: x[0])
type(firstnames)
df['firstnames'] = firstnames
FN = df.groupby('firstnames').count()
type(FN)
NFN = FN['Name']
NFNSORTED = NFN[NFN > 0].sort_values(ascending=False)
NFNSORTED
# # matplotlib.pyplot.hist(NFN[NFN > 1])
#ASTRONAMES = NFNSORTED.to_frame("")
ASTRONAMES = pd.DataFrame({'Astro names':NFNSORTED.index, 'Astro number':NFNSORTED.values})
ASTRONAMES
count = 0
for value in NFN:
    if value > count:
        count = value 
print(count)

usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")
usa_names

query = """SELECT name, sum(number) as number FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY name order by number desc"""
names = usa_names.query_to_pandas_safe(query)
names.to_csv("usa_names.csv")
OVER1M = names[names['number'] > 10000]
OVER1M
# OVER1MN = OVER1M['name']
# OVER1MN
# OVER1MM = OVER1M['number']
OVER1MN = OVER1M['name']
ASTRONAMES['world names'] = OVER1MN
OVER1MM = OVER1M['number']
ASTRONAMES['World number'] = OVER1MM
ASTRONAMES
ASTRO_WITH_TOTAL = ASTRONAMES.set_index('Astro names').join(OVER1M.set_index('name'))
ASTRO_WITH_TOTAL
ASTRO_WITH_TOTAL_NAN = ASTRO_WITH_TOTAL[np.isfinite(ASTRO_WITH_TOTAL['number'])]
ASTRO_WITH_TOTAL_NAN

# ASTRO_WITH_TOTAL_NAN.index.values

# b = ASTRO_WITH_TOTAL_NAN['Astro names'] 
# b

# LS = []
# TN = ASTRO_WITH_TOTAL_NAN['number']
# TN
# for NUM in TN: 
#     ROUND = int(str(NUM)[:2])
#     LS.append(ROUND)
# LS

# for xx in LS

#     xo = (xo-xo) + 3 
#     LS.append(xo)
# LS
# VQ['Astro number']
# VQ = EQ[EQ['number'].map(len) < 2]
LS = []
TN = ASTRO_WITH_TOTAL_NAN['number']
TN
for NUM in TN: 
    ROUND = int(str(NUM)[:2])
    LS.append(ROUND)
LS

ASS = ASTRO_WITH_TOTAL_NAN['Astro number']
ASS

k = [(x / y) * 100 for x, y in zip(ASS, LS)]
k
ALIST = ASTRO_WITH_TOTAL_NAN['Astro number']
SUM0 = sum(ALIST)

WLIST = ASTRO_WITH_TOTAL_NAN['number']
SUM1 = sum(WLIST)

PLIST0 = []
for py in ALIST:
    py = (py / SUM0) * 100
    PLIST0.append(py)
PLIST0

PLIST1 = []
for pi in WLIST:
    pi = (pi / SUM1) * 100
    PLIST1.append(pi)
PLIST1

CHANCECOLLUM = [(x - y) / y for x, y in zip(PLIST0, PLIST1)]
CHANCECOLLUM

NAMECOLLUM = ASTRO_WITH_TOTAL_NAN.index.values
# NAMECOLLUM = ASTRONAMES['Astro names']
# NAMECOLLUM

DAYTA = {'NAME': NAMECOLLUM, 'CHANCE': CHANCECOLLUM }
NAMEANDCHANCE = pd.DataFrame(data=DAYTA)
NAMEANDCHANCE

NAMEANDCHANCESORTED = NAMEANDCHANCE.sort_values(by=['CHANCE'], ascending=False)
NAMEANDCHANCESORTED

# VIBE = pd.DataFrame({'NAME':NAMECOLLUM.index, 'CHANCE':CHANCECOLLUM.values})
# VIBE

dictionary = dict(zip(NAMECOLLUM, CHANCECOLLUM))
dictionary 

INDEX = NAMEANDCHANCESORTED['CHANCE']
INDEX

mp.pyplot.hist(INDEX, bins = range(-1,27))

# matplotlib.pyplot.hist(x = year[INDEX])
# year.sort_values()

# mp.plot(INDEX)
# mp.ylabel('some numbers')
# mp.show()
