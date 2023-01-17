# Data was copied from
# https://covid.ourworldindata.org/data/ecdc/full_data.csv
#
# Question: is there a way for notebook to load date from source so that it is always the updated version? 
# (this notebook uses a 'static' copy)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import urllib
import matplotlib.pyplot as plt
import numpy as np
nr=3 # number of rows to display for each dataframe
# Input data files are available in the "../input/" directory.

#ds = pd.read_csv( "file://rarrieta.byethost7.com/covid19/WHO-200329.csv", header='infer', chunksize=None, encoding=None ).head()
ds = pd.read_csv( "../input/who200411/WHO-200411.csv", verbose=True )
ds['TD_to_TC'] = ds.total_deaths/ds.total_cases
rpt_date='2020-04-11' # string of current report date YYYY-MM-DD
moreThan1000New_cases=ds[(ds.new_cases > 1000) &(ds.location != 'World' ) & (ds.date.str.contains(rpt_date))].sort_values(['new_cases' ], ascending=False).head(nr)
moreThan33New_deaths= ds[(ds.new_deaths > 33) &(ds.location != 'World' ) & (ds.date.str.contains(rpt_date))].sort_values(['new_deaths' ], ascending=False).head(nr)
atHighesttotal_cases= ds[(ds.total_cases > 10000) &(ds.location != 'World' ) & (ds.date.str.contains(rpt_date)) ].sort_values(['total_cases','location'], ascending=False).head(nr)
atHighesttotal_deaths= ds[(ds.total_deaths > 100) &(ds.location != 'World' ) & (ds.date.str.contains(rpt_date)) ].sort_values(['total_deaths','location'], ascending=False).head(nr)

moreThan1000New_cases

moreThan33New_deaths
atHighesttotal_cases
atHighesttotal_deaths
moreThanPt10TD_to_TC= ds[(ds.TD_to_TC > .0005) &(ds.total_deaths > 100) &(ds.location != 'World' ) & (ds.date.str.contains(rpt_date))].sort_values(['TD_to_TC','location'], ascending=False).head(nr)
moreThanPt10TD_to_TC
allTimeHighestNew_cases=ds[(ds.new_cases > 16700) &(ds.location != 'World' ) ].sort_values(['new_cases' ], ascending=False).head(nr)
allTimeHighestNew_cases
allTimeHighestNew_deaths=ds[(ds.new_deaths > 900) &(ds.location != 'World' ) ].sort_values(['new_deaths' ], ascending=False).head(nr)
allTimeHighestNew_deaths.head(10)
allTimeHighestTD_to_TC=ds[(ds.TD_to_TC > .09) & (ds.total_cases > 75000) &(ds.location != 'World' ) ].sort_values(['TD_to_TC' ], ascending=False).head(nr)
allTimeHighestTD_to_TC.head(10)
#.to_string(index=False)

print("Covid-19 Report_date: ",rpt_date," Data: https://covid.ourworldindata.org/data/ecdc/full_data.csv \n\nall Time Highest New_cases\n",allTimeHighestNew_cases.to_string(index=False) ,"\n\nall Time Highest New_deaths\n",allTimeHighestNew_deaths.to_string(index=False),"\n\nall time Highest total_cases\n",atHighesttotal_cases.to_string(index=False),"\n\nall time Highest total_deaths\n",atHighesttotal_deaths.to_string(index=False),"\n\nall Time Highest TD_to_TC (% total deaths TO total cases )\n", allTimeHighestTD_to_TC.to_string(index=False))
plt2 = plt
dp=ds[(ds.location == 'Philippines') & (ds.total_deaths > 12)]
x = dp.date

plt.figure(figsize=(18, 6))
plt.subplot(131)
plt.ylabel('#  o f  P E R S O N S')
plt.xlabel('D A T E')
plt.bar(x, dp.new_cases)
plt.xticks(rotation=90)
plt.subplot(132)
plt.bar(x, dp.new_deaths  )
plt.xticks(rotation=90)
plt.subplot(133)
plt.plot(x, dp.total_cases, 'r--',x,dp.total_deaths, 'b--')
plt.xticks(rotation=90)
plt.suptitle('PH Covid-19 NEW CASES                                                        NEW DEATHS                                                     TOTAL CASES & TOTAL DEATHS     ')
plt.show()

#dp.sort_values(['date'],ascending=False)

di=ds[(ds.location == 'Canada') & (ds.total_deaths > 1)] #.head(34)
x = di.date

plt2.figure(figsize=(18, 6))
plt2.subplot(131)
plt2.ylabel('#  o f  P E R S O N S')
plt2.xlabel('D A T E')
plt2.bar(x, di.new_cases)
plt2.xticks(rotation=90)
plt2.subplot(132)
plt2.bar(x, di.new_deaths  )
plt2.xticks(rotation=90)
plt2.subplot(133)
plt2.plot(x, di.total_cases, 'r--',x,di.total_deaths, 'b--')
plt2.xticks(rotation=90)
plt2.suptitle('      Covid-19 NEW CASES                                                        NEW DEATHS                                                     TOTAL CASES & TOTAL DEATHS     ')
plt2.show()
ds[(ds.total_deaths == 0)& (ds.date.str.contains(rpt_date))]