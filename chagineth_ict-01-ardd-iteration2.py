# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/ardd-fatal-crashes-march/ardd_fatal_crashes_march.csv', index_col=['CrashID']) 

ds.head(10)
col = ds.columns

print(col)
# Graph 01- Yearly Fatal Crashes per 10,000 People



# Graph changed based on 06/05/2020 Dmitry's review comments- iteration-2

# Removed year 2020 details  



data = {'Year': ["1989", "1990", "1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                "2001", "2002", "2003", "2004","2005","2006", "2007","2008","2009","2010",

                "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[2407,2050,1874,1736,1737,1702,1822,1768,1601,1573,1553,1628,

                              1584,1525,1445,1444,1472,1452,1453,1315,1347,1233,

                              1151,1190,1101,1051,1100,1198,1125,1055,1108],

        'tot_population':[16936723,17169768,17378981,17557133,17719090,17893433,18119616,18330079,18510004,18705620,18919210,

                19141036,19386461,19605441,19827155,20046003,20311543,20627547,21016121,21475625,21865623,22172469,

                22522197,22928023,23297777,23640331,23984581,24389684,24773350,25171439,25464116]}



Yearly_Crashes_ds = pd.DataFrame.from_dict(data)

# print(new_ds)



Yearly_Crashes_ds['per 10000 ppl'] = ((Yearly_Crashes_ds['tot # of Accidents'] / Yearly_Crashes_ds['tot_population']) * 10000).round(2)

print(Yearly_Crashes_ds)



Yearly_Crashes_ds.plot(kind='bar', x='Year',y='per 10000 ppl',ylim =(0.0,1.6), color='green')

plt.title("Yearly Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
# Graph 02- Each State # of Fatal Crashes per 10,000 People - from 1989-2020



# Graph changed based on 06/05/2020 Dmitry's review comments- iteration-2

# Have to take # of Fatal Crashes for 1 Year base for 10,000 ppl

# so divided each state tot # of accidenents value by 30 (1989-2020)



data = {'State': ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"], 

        'tot # of Accidents':[14260,10161,9081,5459,3918,1336,1383,424],

        'tot_population':[8117976,6629870,5115451,2630557,1756494,535500,245562,428060]}



state_crashes_ds = pd.DataFrame.from_dict(data)

# print(state_crashes_ds)



state_crashes_ds['per 10000 ppl'] = ((((state_crashes_ds['tot # of Accidents'])/30) / state_crashes_ds['tot_population']) * 10000).round(2)

print(state_crashes_ds)



state_crashes_ds.plot(kind='bar', x='State',y='per 10000 ppl', color='pink', ylim=(0,2))

plt.title("Each State Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
# Graph 03 - % of Fatal Crashes based on Speed Limit for 1989- 2020



# Graph changed based on 06/05/2020 Dmitry's review comments- iteration-2

# Have to take # of Fatal Crashes for two speed limit category <70 and >70 (2 bins)



# Iteration-1 graph (old)

# data = {'SpeedLimit': ["<=40","41-50","51-60","61-70","71-80","81-90","91-100","101-110",">110"], 

#         'tot # of Accidents':[373,2598,12698,2493,5351,971,15194,4962,87]}



# Iteration-2 graph (new)

data = {'SpeedLimit': ["<=70", ">70"], 

        'tot # of Accidents':[18162,26565]}



speed_crash_ds = pd.DataFrame.from_dict(data)

# print(speed_crash_ds)



speed_crash_ds['Crash %'] = ((speed_crash_ds['tot # of Accidents'] / speed_crash_ds['tot # of Accidents'].sum()) * 100).round(2)

print(speed_crash_ds)



speed_crash_ds.plot(kind='bar', x='SpeedLimit',y='Crash %', color='brown', ylim=(0,100))

plt.title("% of Fatal Crashes based on Speed Limit", y=1.05);

plt.ylabel("% of Fatal Crashes", labelpad=14)
# Get year by year (2011 - 2020) total # of Crashes in each state for Graph 04

filteredData = ds.Year == 2020

print ((ds[filteredData])['State'].value_counts())

filteredData = ds.Year == 2019

print ((ds[filteredData])['State'].value_counts())

filteredData = ds.Year == 2018

print((ds[filteredData])['State'].value_counts())

filteredData = ds.Year == 2017

print((ds[filteredData])['State'].value_counts())

filteredData = ds.Year == 2016

print((ds[filteredData])['State'].value_counts())

filteredData = ds.Year == 2015

print((ds[filteredData])['State'].value_counts())

filteredData = ds.Year == 2014

print((ds[filteredData])['State'].value_counts())

filteredData = ds.Year == 2013

print((ds[filteredData])['State'].value_counts())

filteredData = ds.Year == 2012

print((ds[filteredData])['State'].value_counts())

filteredData = ds.Year == 2011

print((ds[filteredData])['State'].value_counts())
# Get year by year (2011 - 2020) total # of Crashes in each state per 10000 ppl for Graph 04



# For year 2020

data = {'State': ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"], 

        'tot # of Accidents':[46,40,24,20,15,10,4,0],

        'tot_population':[8117976,6629870,5115451,2630557,1756494,535500,245562,428060]}

crashes2020_ds = pd.DataFrame.from_dict(data)



crashes2020_ds['per 10000 ppl'] = ((crashes2020_ds['tot # of Accidents'] / crashes2020_ds['tot_population']) * 10000).round(2)

print(crashes2020_ds)



# For year 2019

data = {'State': ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"], 

        'tot # of Accidents':[331,252,195,154,110,32,28,6],

        'tot_population':[8117976,6629870,5115451,2630557,1756494,535500,245562,428060]}

crashes2019_ds = pd.DataFrame.from_dict(data)

# print(crashes2019_ds)



crashes2019_ds['per 10000 ppl'] = ((crashes2019_ds['tot # of Accidents'] / crashes2019_ds['tot_population']) * 10000).round(2)

print(crashes2019_ds)



# For year 2018

data = {'State': ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"], 

        'tot # of Accidents':[326,224,202,145,75,42,32,9],

        'tot_population':[8038400,6528614,5050116,2605636,1743284,531833,245583,423329]}

crashes2018_ds = pd.DataFrame.from_dict(data)



crashes2018_ds['per 10000 ppl'] = ((crashes2018_ds['tot # of Accidents'] / crashes2018_ds['tot_population']) * 10000).round(2)

print(crashes2018_ds)



# For year 2017

data = {'State': ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"], 

        'tot # of Accidents':[351,240,228,151,93,30,27,5],

        'tot_population':[7919815,6387081,4963072,2582563,1728494,524969,246858,415874]}

crashes2017_ds = pd.DataFrame.from_dict(data)



crashes2017_ds['per 10000 ppl'] = ((crashes2017_ds['tot # of Accidents'] / crashes2017_ds['tot_population']) * 10000).round(2)

print(crashes2017_ds)



# For year 2016

data = {'State': ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"], 

        'tot # of Accidents':[356,275,238,170,76,40,33,10],

        'tot_population':[7801785,6244863,4883821,2563708,1717400,519810,246183,407489]}

crashes2016_ds = pd.DataFrame.from_dict(data)



crashes2016_ds['per 10000 ppl'] = ((crashes2016_ds['tot # of Accidents'] / crashes2016_ds['tot_population']) * 10000).round(2)

print(crashes2016_ds)



# For year 2015

data = {'State': ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"], 

        'tot # of Accidents':[326,231,219,140,96,42,32,14],

        'tot_population':[7671401,6093049,4804933,4804933,1705937,515694,244090,398874]}

crashes2015_ds = pd.DataFrame.from_dict(data)



crashes2015_ds['per 10000 ppl'] = ((crashes2015_ds['tot # of Accidents'] / crashes2015_ds['tot_population']) * 10000).round(2)

print(crashes2015_ds)



# For year 2014

data = {'State': ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"], 

        'tot # of Accidents':[285,223,199,173,96,34,31,10],

        'tot_population':[7562171,5957512,4747263,2528619,1693107,514040,242753,391981]}

crashes2014_ds = pd.DataFrame.from_dict(data)



crashes2014_ds['per 10000 ppl'] = ((crashes2014_ds['tot # of Accidents'] / crashes2014_ds['tot_population']) * 10000).round(2)

print(crashes2014_ds)



# For year 2013

data = {'State': ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"], 

        'tot # of Accidents':[316,246,225,149,90,35,33,7],

        'tot_population':[7454938,5832585,4685439,2502188,1678052,513015,242304,386318]}

crashes2013_ds = pd.DataFrame.from_dict(data)



crashes2013_ds['per 10000 ppl'] = ((crashes2013_ds['tot # of Accidents'] / crashes2013_ds['tot_population']) * 10000).round(2)

print(crashes2013_ds)



# For year 2012

data = {'State': ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"], 

        'tot # of Accidents':[336,261,255,171,86,40,29,12],

        'tot_population':[7353189,5709586,4611304,2457489,1663082,511813,238728,379812]}

crashes2012_ds = pd.DataFrame.from_dict(data)



crashes2012_ds['per 10000 ppl'] = ((crashes2012_ds['tot # of Accidents'] / crashes2012_ds['tot_population']) * 10000).round(2)

print(crashes2012_ds)



# For year 2011

data = {'State': ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"], 

        'tot # of Accidents':[336,259,227,167,95,38,23,6],

        'tot_population':[7258722,5591818,4518649,2385947,1647183,511739,232952,372070]}

crashes2011_ds = pd.DataFrame.from_dict(data)



crashes2011_ds['per 10000 ppl'] = ((crashes2011_ds['tot # of Accidents'] / crashes2011_ds['tot_population']) * 10000).round(2)

print(crashes2011_ds)
# Graph 4 -  Each Year (2011 - 2020) each State Fatal Crashes per 10,000 People



# Graph changed based on 06/05/2020 Dmitry's review comments- iteration-2

# Have to take # of Fatal Crashes line graphs for each Year from 2010 to 2019





import matplotlib.pyplot as plt



year2011 = [0.46,0.46,0.50,0.70,0.58,0.74,0.99,0.16]

year2012 = [0.46,0.46,0.55,0.70,0.52,0.78,1.21,0.32]

year2013 = [0.42,0.42,0.48,0.60,0.54,0.68,1.36,0.18]

year2014 = [0.38,0.37,0.42,0.68,0.57,0.66,1.28,0.26]

year2015 = [0.42,0.38,0.46,0.29,0.56,0.81,1.31,0.35]

year2016 = [0.46,0.44,0.49,0.66,0.44,0.77,1.34,0.25]

year2017 = [0.44,0.38,0.46,0.58,0.54,0.57,1.09,0.12]

year2018 = [0.41,0.34,0.40,0.56,0.43,0.79,1.30,0.21]

year2019 = [0.41,0.38,0.38,0.59,0.63,0.60,1.14,0.14]

year2020 = [0.06,0.06,0.05,0.08,0.09,0.19,0.16,0.00]

index = ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"]

df = pd.DataFrame({'2011': year2011,'2012': year2012,'2013': year2013,'2014': year2014,'2015': year2015,

                   '2016': year2016,'2017': year2017,'2018': year2018,'2019': year2019,'2020': year2020}, index=index)

ax = df.plot.line(rot=90)



plt.title("2011 - 2020 Each State Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)

plt.xlabel("State", labelpad=14)
# Get each state total # of Crashes in every year (1989-2020) for Graph 04

filteredData = ds.State == 'NSW'

print ((ds[filteredData])['Year'].value_counts())

# 1989- 2020 (784,702,585,576,518,552,563,538,525,491,506,543,

#                     486,501,483,458,459,449,405,353,409,365, 

#                     336,336,316,285,326,356,351,326,331,46)

filteredData1 = ds.State == 'Vic'

print ((ds[filteredData1])['Year'].value_counts())

# 1989- 2020 (681,492,435,365,381,345,371,382,346,348,345,373,

#                     404,361,294,312,314,309,289,278,268,260, 

#                     259,261,225,223,231,275,240,202,252,40)

filteredData = ds.State == 'Qld'

print ((ds[filteredData])['Year'].value_counts())

# 1989- 2020 (376,347,359,363,357,364,408,338,321,257,273,275,

#                     296,283,284,289,296,313,338,294,296,236, 

#                     227,255,246,199,219,238,228,224,195,24)

filteredData = ds.State == 'WA'

print ((ds[filteredData])['Year'].value_counts())

# 1989- 2020 (214,181,187,171,191,195,194,220,184,199,189,184,

#                     151,159,155,162,151,181,214,185,176,176,

#                     167,171,149,173,140,170,151,145,154,20)

filteredData = ds.State == 'SA'

print ((ds[filteredData])['Year'].value_counts())

# 1989- 2020 (201,187,166,142,191,143,163,162,123,152,132,151,

#                     137,138,136,128,127,104,107,87,104,105,

#                      95,86,90,96,96,76,93,75,110,15)

filteredData = ds.State == 'Tas'

print ((ds[filteredData])['Year'].value_counts())

# 1989- 2020 (68,63,66,59,47,52,53,53,29,47,47,38,

#                   52,35,39,52,49,43,39,37,52,29,

#                   23,29,35,31,32,33,30,32,32,10)

filteredData = ds.State == 'NT'

print ((ds[filteredData])['Year'].value_counts())

# 1989- 2020 (57,54,60,42,41,36,56,58,56,59,44,48,

#                   43,40,44,34,51,41,47,67,31,46,

#                   38,40,33,34,42,40,27,42,28,4)

filteredData = ds.State == 'ACT'

print ((ds[filteredData])['Year'].value_counts())

# 1989- 2020 (26,24,16,18,11,15,14,17,17,20,17,16,

#                   15,08,10,09,25,12,07,14,11,16,

#                   6,12,07,10,14,10,05,09,06,00)
# Graph 04 - SUB 01 - NSW (graph 4 need to divide 8 different graphs based on state)

# Get each state total # of Crashes in each year(1989-2020) per 10000 ppl for Graph 04



# Graph changed based on 13/05/2020 Dmitry's review comments- iteration-2

# Have to take # of Fatal Crashes line graphs for one state(NSW) each Year from 1989 to 2019



# For State NSW

data = {'Year': ["1989","1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                 "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",

                 "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[784,702,585,576,518,552,563,538,525,491,506,543,

                              486,501,483,458,459,449,405,353,409,365, 

                              336,336,316,285,326,356,351,326,331],

        'tot_population':[5803079,5862497,5928072,5977823,6020171,6071872,6143971,6214548,6274966,6338790,6409971,6485081,

                          6558484,6599441,6634509,6669206,6718023,6786160,6883852,7001782,7101504,7179891,

                          7258722,7353189,7454938,7562171,7671401,7801785,7919815,8038400,8117976]}

crashesNSW_ds = pd.DataFrame.from_dict(data)



crashesNSW_ds['per 10000 ppl'] = ((crashesNSW_ds['tot # of Accidents'] / crashesNSW_ds['tot_population']) * 10000).round(2)

print(crashesNSW_ds)



crashesNSW_ds.plot(kind='line', x='Year',y='per 10000 ppl', color='pink', ylim=(0,1.5))

plt.title("NSW State Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
# Graph 04 - SUB 02 - VIC (graph 4 need to divide 8 different graphs based on state)

# Get each state total # of Crashes in each year(1989-2020) per 10000 ppl for Graph 04



# Graph changed based on 13/05/2020 Dmitry's review comments- iteration-2

# Have to take # of Fatal Crashes line graphs for one state(VIC) each Year from 1989 to 2019



# For State VIC

data = {'Year': ["1989","1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                 "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",

                 "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[681,492,435,365,381,345,371,382,346,348,345,373,

                              404,361,294,312,314,309,289,278,268,260, 

                              259,261,225,223,231,275,240,202,252],

        'tot_population':[4348225,4400707,4435083,4458219,4466738,4483205,4517353,4552904,4586156,4629345,4677581,4730855,

                          4790212,4845024,4900176,4957147,5023203,5103965,5199503,5313285,5419249,5495711,

                          5591818,5709586,5832585,5957512,6093049,6244863,6387081,6528614,6629870]}

crashesVIC_ds = pd.DataFrame.from_dict(data)



crashesVIC_ds['per 10000 ppl'] = ((crashesVIC_ds['tot # of Accidents'] / crashesVIC_ds['tot_population']) * 10000).round(2)

print(crashesVIC_ds)



crashesVIC_ds.plot(kind='line', x='Year',y='per 10000 ppl', color='purple', ylim=(0,1.7))

plt.title("VIC State Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
# Graph 04 - SUB 03 - QLD (graph 4 need to divide 8 different graphs based on state)

# Get each state total # of Crashes in each year(1989-2020) per 10000 ppl for Graph 04



# Graph changed based on 13/05/2020 Dmitry's review comments- iteration-2

# Have to take # of Fatal Crashes line graphs for one state(QLD) each Year from 1989 to 2019



# For State QLD

data = {'Year': ["1989","1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                 "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",

                 "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[376,347,359,363,357,364,408,338,321,257,273,275,

                              296,283,284,289,296,313,338,294,296,236, 

                              227,255,246,199,219,238,228,224,195],

        'tot_population':[2864007,2928713,2990441,3057138,3130986,3198877,3271743,3330579,3380394,3427505,3481034,3537670,

                          3611203,3700791,3788560,3872351,3964175,4055845,4159990,4275551,4367454,4436882,

                          4518649,4611304,4685439,4747263,4804933,4883821,4963072,5050116,5115451]}

crashesQLD_ds = pd.DataFrame.from_dict(data)



crashesQLD_ds['per 10000 ppl'] = ((crashesQLD_ds['tot # of Accidents'] / crashesQLD_ds['tot_population']) * 10000).round(2)

print(crashesQLD_ds)



crashesQLD_ds.plot(kind='line', x='Year',y='per 10000 ppl', color='purple', ylim=(0,1.5))

plt.title("QLD State Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
# Graph 04 - SUB 04 - WA (graph 4 need to divide 8 different graphs based on state)

# Get each state total # of Crashes in each year(1989-2020) per 10000 ppl for Graph 04



# Graph changed based on 13/05/2020 Dmitry's review comments- iteration-2

# Have to take # of Fatal Crashes line graphs for one state(WA) each Year from 1989 to 2019



# For State WA

data = {'Year': ["1989","1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                 "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",

                 "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[214,181,187,171,191,195,194,220,184,199,189,184,

                              151,159,155,162,151,181,214,185,176,176,

                              167,171,149,173,140,170,151,145,154],

        'tot_population':[1596225,1624390,1647408,1668515,1690348,1718549,1751933,1783556,1810928,1840078,1866265,1892531,

                          1917752,1938610,1966130,1994241,2029936,2076867,2135006,2208928,2263747,2319063,

                          2385947,2457489,2502188,2528619,2547745,2563708,2582563,2605636,2630557]}

crashesWA_ds = pd.DataFrame.from_dict(data)



crashesWA_ds['per 10000 ppl'] = ((crashesWA_ds['tot # of Accidents'] / crashesWA_ds['tot_population']) * 10000).round(2)

print(crashesWA_ds)



crashesWA_ds.plot(kind='line', x='Year',y='per 10000 ppl', color='purple', ylim=(0,1.5))

plt.title("WA State Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
# Graph 04 - SUB 05 - SA (graph 4 need to divide 8 different graphs based on state)

# Get each state total # of Crashes in each year(1989-2020) per 10000 ppl for Graph 04



# Graph changed based on 13/05/2020 Dmitry's review comments- iteration-2

# Have to take # of Fatal Crashes line graphs for one state(SA) each Year from 1989 to 2019



# For State SA

data = {'Year': ["1989","1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                 "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",

                 "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[201,187,166,142,191,143,163,162,123,152,132,151,

                              137,138,136,128,127,104,107,87,104,105,

                               95,86,90,96,96,76,93,75,110],

        'tot_population':[1425461,1438882,1450862,1457241,1461102,1463977,1466605,1471997,1479003,1487042,1495218,1500129,

                          1507825,1515723,1524727,1532562,1544852,1561300,1578489,1597880,1618578,1632482,

                          1647183,1663082,1678052,1693107,1705937,1717400,1728494,1743284,1756494]}

crashesSA_ds = pd.DataFrame.from_dict(data)



crashesSA_ds['per 10000 ppl'] = ((crashesSA_ds['tot # of Accidents'] / crashesSA_ds['tot_population']) * 10000).round(2)

print(crashesSA_ds)



crashesSA_ds.plot(kind='line', x='Year',y='per 10000 ppl', color='purple', ylim=(0,1.6))

plt.title("SA State Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
# Graph 04 - SUB 06 - TAS (graph 4 need to divide 8 different graphs based on state)

# Get each state total # of Crashes in each year(1989-2020) per 10000 ppl for Graph 04



# Graph changed based on 13/05/2020 Dmitry's review comments- iteration-2

# Have to take # of Fatal Crashes line graphs for one state(TAS) each Year from 1989 to 2019



# For State TAS

data = {'Year': ["1989","1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                 "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",

                 "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[68,63,66,59,47,52,53,53,29,47,47,38,

                              52,35,39,52,49,43,39,37,52,29,

                              23,29,35,31,32,33,30,32,32],

        'tot_population':[458410,464520,468549,471258,472983,474076,475148,475529,474215,473450,473294,473200,

                          473890,475998,481411,484778,488098,491515,495858,501774,506461,510219,

                          511739,511813,513015,514040,515694,519810,524969,531833,535500]}

crashesTAS_ds = pd.DataFrame.from_dict(data)



crashesTAS_ds['per 10000 ppl'] = ((crashesTAS_ds['tot # of Accidents'] / crashesTAS_ds['tot_population']) * 10000).round(2)

print(crashesTAS_ds)



crashesTAS_ds.plot(kind='line', x='Year',y='per 10000 ppl', color='purple', ylim=(0,1.6))

plt.title("TAS State Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
# Graph 04 - SUB 07 - NT (graph 4 need to divide 8 different graphs based on state)

# Get each state total # of Crashes in each year(1989-2020) per 10000 ppl for Graph 04



# Graph changed based on 13/05/2020 Dmitry's review comments- iteration-2

# Have to take # of Fatal Crashes line graphs for one state(NT) each Year from 1989 to 2019



# For State NT

data = {'Year': ["1989","1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                 "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",

                 "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[57,54,60,42,41,36,56,58,56,59,44,48,

                              43,40,44,34,51,41,47,67,31,46,

                              38,40,33,34,42,40,27,42,28],

        'tot_population':[162097,165047,167043,170420,173590,176761,182829,187342,191259,194390,197757,200045,

                          201751,201549,201708,203857,207385,211029,216618,222526,227783,230299,

                          232952,238728,242304,242753,244090,246183,246858,245583,245562]}

crashesNT_ds = pd.DataFrame.from_dict(data)



crashesNT_ds['per 10000 ppl'] = ((crashesNT_ds['tot # of Accidents'] / crashesNT_ds['tot_population']) * 10000).round(2)

print(crashesNT_ds)



crashesNT_ds.plot(kind='line', x='Year',y='per 10000 ppl', color='purple', ylim=(0,4.0))

plt.title("NT State Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
# Graph 04 - SUB 08 - ACT (graph 4 need to divide 8 different graphs based on state)

# Get each state total # of Crashes in each year(1989-2020) per 10000 ppl for Graph 04



# Graph changed based on 13/05/2020 Dmitry's review comments- iteration-2

# Have to take # of Fatal Crashes line graphs for one state(ACT) each Year from 1989 to 2019



# For State ACT

data = {'Year': ["1989","1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                 "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",

                 "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[26,24,16,18,11,15,14,17,17,20,17,16,

                              15,8,10,9,25,12,7,14,11,16,

                              6,12,7,10,14,10,5,9,6],

        'tot_population':[279219,285012,291523,296519,300490,303289,307022,310655,310281,312300,315431,318941,

                          322874,325950,327596,329498,333505,338381,344176,351101,357859,364833,

                          372070,379812,386318,391981,398874,407489,415874,423329,428060]}

crashesACT_ds = pd.DataFrame.from_dict(data)



crashesACT_ds['per 10000 ppl'] = ((crashesACT_ds['tot # of Accidents'] / crashesACT_ds['tot_population']) * 10000).round(2)

print(crashesACT_ds)



crashesACT_ds.plot(kind='line', x='Year',y='per 10000 ppl', color='purple', ylim=(0,1.1))

plt.title("ACT State Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
# Graph 03 - % of Fatal Crashes based on Speed Limit for 1989- 2020



# Graph changed based on 06/05/2020 Dmitry's review comments- iteration-2

# Have to take # of Fatal Crashes for two speed limit category <70 and >70 (2 bins)



# Iteration-1 graph (old)

# data = {'SpeedLimit': ["<=40","41-50","51-60","61-70","71-80","81-90","91-100","101-110",">110"], 

#         'tot # of Accidents':[373,2598,12698,2493,5351,971,15194,4962,87]}



# Iteration-2 graph (new)

# data = {'SpeedLimit': ["<=70", ">70"], 

#         'tot # of Accidents':[18162,26565]}



# Iteration-2 graph (new-2)

#speed >70 for year wise (1989 - 2020)

data = {'Year': ["1989","1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                 "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",

                "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[1256,1116,1047,978,946,948,1026,948,909,909,901,937,

                               878,915,857,846,881,851,897,776,838,793,

                               717,705,679,648,680,774,662,641,702]}



speed_crash_ds = pd.DataFrame.from_dict(data)

# print(speed_crash_ds)



speed_crash_ds['Crash %'] = ((speed_crash_ds['tot # of Accidents'] / speed_crash_ds['tot # of Accidents'].sum()) * 100).round(2)

print(speed_crash_ds)



speed_crash_ds.plot(kind='line', x='Year',y='Crash %', color='brown', ylim=(0,5))

plt.title("% of Fatal Crashes happen in Speed Limit > 70", y=1.05);

plt.ylabel("% of Fatal Crashes", labelpad=14)
# Graph 03 - % of Fatal Crashes based on Speed Limit for 1989- 2020



# Iteration-2 graph (new-2)

#speed <=70 for year wise (1989 - 2020)

data = {'Year': ["1989","1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                 "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",

                 "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[1077,865,761,691,719,709,741,759,637,605,613,633,

                              619,557,543,556,529,561,512,521,496,427,

                              412,461,405,385,408,436,452,402,380]}



speed_crash_ds = pd.DataFrame.from_dict(data)

# print(speed_crash_ds)



speed_crash_ds['Crash %'] = ((speed_crash_ds['tot # of Accidents'] / speed_crash_ds['tot # of Accidents'].sum()) * 100).round(2)

print(speed_crash_ds)



speed_crash_ds.plot(kind='line', x='Year',y='Crash %', color='brown', ylim=(0,7))

plt.title("% of Fatal Crashes happen in Speed Limit <=70", y=1.05);

plt.ylabel("% of Fatal Crashes", labelpad=14)
# Graph 4 -  Each state Crashes per 10,000 People for years (1989 - 2019)



# Graph changed based on 20/05/2020 Dmitry's review comments- iteration-2

# Graph hanges to years vs.10,000 People





import matplotlib.pyplot as plt



NSW = [1.35,1.20,0.99,0.96,0.86,0.91,0.92,0.87,0.84,0.77,0.79,0.84,0.74,0.76,0.73,0.69,0.68,0.66,0.59,0.50,0.58,0.51,0.46,0.46,0.42,0.38,0.42,0.46,0.44,0.41,0.41]

VIC = [1.57,1.12,0.98,0.82,0.85,0.77,0.82,0.84,0.75,0.75,0.74,0.79,0.84,0.75,0.60,0.63,0.63,0.61,0.56,0.52,0.49,0.47,0.46,0.46,0.39,0.37,0.38,0.44,0.38,0.31,0.38]

QLD = [1.31,1.18,1.20,1.19,1.14,1.14,1.25,1.01,0.95,0.75,0.78,0.78,0.82,0.76,0.75,0.75,0.75,0.77,0.81,0.69,0.68,0.53,0.50,0.55,0.53,0.42,0.46,0.49,0.46,0.44,0.38]

WA  = [1.34,1.11,1.14,1.02,1.13,1.13,1.11,1.23,1.02,1.08,1.01,0.97,0.79,0.82,0.79,0.81,0.74,0.87,1.00,0.84,0.78,0.76,0.70,0.70,0.60,0.68,0.55,0.66,0.58,0.56,0.59]

SA  = [1.41,1.30,1.14,0.97,1.31,0.98,1.11,1.10,0.83,1.02,0.88,1.01,0.91,0.91,0.89,0.84,0.82,0.67,0.68,0.54,0.64,0.64,0.58,0.52,0.54,0.57,0.56,0.44,0.54,0.43,0.63]

TAS = [1.48,1.36,1.41,1.25,0.99,1.10,1.12,1.11,0.61,0.99,0.99,0.80,1.10,0.74,0.81,1.07,1.00,0.87,0.79,0.74,1.03,0.57,0.45,0.57,0.68,0.60,0.62,0.63,0.57,0.60,0.60]

NT  = [3.52,3.27,3.59,2.46,2.36,2.04,3.06,3.10,2.93,3.04,2.22,2.40,2.13,1.98,2.18,1.67,2.46,1.94,2.17,3.01,1.36,2.00,1.63,1.68,1.36,1.40,1.72,1.62,1.09,1.71,1.14]

ACT = [0.93,0.84,0.55,0.61,0.37,0.49,0.46,0.55,0.55,0.64,0.54,0.50,0.46,0.25,0.31,0.27,0.75,0.35,0.20,0.40,0.31,0.44,0.16,0.32,0.18,0.26,0.35,0.25,0.12,0.21,0.14]



index = ["1989","1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                 "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",

                 "2011","2012","2013","2014","2015","2016","2017","2018","2019"]

df = pd.DataFrame({'NSW': NSW,'VIC': VIC,'QLD': QLD,'WA': WA,'SA': SA,

                   'TAS': TAS,'NT': NT,'ACT': ACT}, index=index)

ax = df.plot.line(rot=90)



plt.title("Each State Fatal Crashes per 10,000 People in years 1989 - 2019", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)

plt.xlabel("State", labelpad=14)