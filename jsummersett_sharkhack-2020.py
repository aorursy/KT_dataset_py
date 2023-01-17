# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



%matplotlib inline



import matplotlib

import numpy as np # linear algebra

import matplotlib.pyplot as plt

import statistics as statistics



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.ticker as ticker



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



covid_data = pd.read_csv('/kaggle/input/usa-covid19-casedeaths-by-county/covid.csv',index_col ="county")

population_data = pd.read_csv('/kaggle/input/usa-population-density-by-state-19102010/pop_density.csv', index_col ="STATE_OR_REGION")
st_den = population_data[['2010_POPULATION','2010_DENSITY', '2010_RANK']]



st_den = st_den.sort_values(by=['2010_RANK'])





# TOP 5 : 'District of Columbia', 'New Jersey' , 'Puerto Rico', 'Rhode Island', 'Massachusetts'

# BOTTOM 5: 'South Dakota', 'North Dakota', 'Montana', 'Wyoming', 'Alaska'

high_dens = st_den.loc[['District of Columbia', 'New Jersey' , 'Puerto Rico', 'Rhode Island', 'Massachusetts']]

low_dens = st_den.loc[['South Dakota', 'North Dakota', 'Montana', 'Wyoming', 'Alaska']]



positive_tests = covid_data[['state', 'date', 'tstpos', 'pbpos']]

positive_tests.head()



dc_data = covid_data.loc["District of Columbia, District of Columbia"]

#DC positive rates by day in chronological order

dc_pos = dc_data.tstpos



nj_data = covid_data.loc["New Jersey (state)"]

#NJ positive rates by day in chronological order

nj_pos = nj_data.tstpos



pr_data = covid_data.loc["Puerto Rico"]

#PR positive rates by day in chronological order

pr_pos = pr_data.tstpos



ri_data = covid_data.loc["Rhode Island (state)"]

#RI positive rates by day in chronological order

ri_pos = ri_data.tstpos



ma_data = covid_data.loc["Massachusetts (state)"]

#MA positive rates by day in chronological order

ma_pos = ma_data.tstpos



#'South Dakota', 'North Dakota', 'Montana', 'Wyoming', 'Alaska']

#SD

sd_data = covid_data.loc["South Dakota (state)"]

sd_pos = sd_data.tstpos



#ND

nd_data = covid_data.loc["North Dakota (state)"]

nd_pos = nd_data.tstpos



#MT

mt_data = covid_data.loc["Montana (state)"]

mt_pos = mt_data.tstpos



#WY

wy_data = covid_data.loc["Wyoming (state)"]

wy_pos = wy_data.tstpos



#al

al_data = covid_data.loc["Alaska (state)"]

al_pos = al_data.tstpos



#list of all dates to use as the x axis in graphs

dates = dc_data.date.tolist()
#infection rate (positive-positive from the day before)/population to get what percent of the population got covid/tested positive that day

#DC

dc_popdens = high_dens.loc['District of Columbia', '2010_POPULATION']

dc_rateslist = [0]

for i in range (1, len(dc_data)):

    dc_rateslist.append(((dc_pos.iloc[i]- dc_pos.iloc[i-1])/dc_popdens))



#NJ

nj_popdens = high_dens.loc['New Jersey', '2010_POPULATION']

nj_rateslist = [0]

for i in range (1, len(nj_data)):

    nj_rateslist.append(((nj_pos.iloc[i]- nj_pos.iloc[i-1])/nj_popdens))



#PR

pr_popdens = high_dens.loc['Puerto Rico', '2010_POPULATION']

pr_rateslist = [0]

for i in range (1, len(pr_data)):

    pr_rateslist.append(((pr_pos.iloc[i]- pr_pos.iloc[i-1])/pr_popdens))

    

#RI

ri_popdens = high_dens.loc['Rhode Island', '2010_POPULATION']

ri_rateslist = [0]

for i in range (1, len(ri_data)):

       ri_rateslist.append(((ri_pos.iloc[i]- ri_pos.iloc[i-1])/ri_popdens))



#MA

ma_popdens = high_dens.loc['Massachusetts', '2010_POPULATION']

ma_rateslist = [0]

for i in range (1, len(ma_data)):

    ma_rateslist.append(((ma_pos.iloc[i]- ma_pos.iloc[i-1])/ma_popdens))

    

    

    

#'South Dakota', 'North Dakota', 'Montana', 'Wyoming', 'Alaska']

#SD

sd_popdens = low_dens.loc['South Dakota', '2010_POPULATION']

sd_rateslist = [0]

for i in range (1, len(sd_data)):

    sd_rateslist.append(((sd_pos.iloc[i]- sd_pos.iloc[i-1])/sd_popdens))

    

#ND

nd_popdens = low_dens.loc['North Dakota', '2010_POPULATION']

nd_rateslist = [0]

for i in range (1, len(nd_data)):

    nd_rateslist.append(((nd_pos.iloc[i]- nd_pos.iloc[i-1])/nd_popdens))

    

#MT

mt_popdens = low_dens.loc['Montana', '2010_POPULATION']

mt_rateslist = [0]

for i in range (1, len(mt_data)):

    mt_rateslist.append(((mt_pos.iloc[i]- mt_pos.iloc[i-1])/mt_popdens))

    

#WY

wy_popdens = low_dens.loc['Wyoming', '2010_POPULATION']

wy_rateslist = [0]

for i in range (1, len(wy_data)):

    wy_rateslist.append(((wy_pos.iloc[i]- wy_pos.iloc[i-1])/wy_popdens))

    

#AL

al_popdens = low_dens.loc['Alaska', '2010_POPULATION']

al_rateslist = [0]

for i in range (1, len(al_data)):

    al_rateslist.append(((al_pos.iloc[i]- al_pos.iloc[i-1])/al_popdens))
mr_dc = statistics.mean(dc_rateslist)

mr_nj = statistics.mean(nj_rateslist)

mr_pr = statistics.mean(pr_rateslist)

mr_ri = statistics.mean(ri_rateslist)

mr_ma = statistics.mean(ma_rateslist)



mr_sd = statistics.mean(sd_rateslist)

mr_nd = statistics.mean(nd_rateslist)

mr_mt = statistics.mean(mt_rateslist)

mr_wy = statistics.mean(wy_rateslist)

mr_al = statistics.mean(al_rateslist)
fig, ax1 = plt.subplots(figsize = (10,6)) 

nyPlot = plt

nyPlot.plot(dates, dc_rateslist)

nyPlot.plot(dates, nj_rateslist)

nyPlot.plot(dates, pr_rateslist)

nyPlot.plot(dates, ri_rateslist)

nyPlot.plot(dates, ma_rateslist)



ax1.xaxis.set_major_locator(ticker.MultipleLocator(60))

nyPlot.title("Percentage of State's Population Testing Positive Each Day (Most Dense)")

nyPlot.legend(['DC','New Jersey' , 'Puerto Rico', 'Rhode Island', 'Massachusetts'])

fig.savefig('topdens_cases.png', dpi=150)

nyPlot.show
#covid data for each state as a vector here:

#military = nyData.c92.tolist()



fig, ax1 = plt.subplots(figsize = (10,6)) 

casePlot_low = plt

casePlot_low.plot(dates, sd_rateslist)

casePlot_low.plot(dates, nd_rateslist)

casePlot_low.plot(dates, mt_rateslist)

casePlot_low.plot(dates, wy_rateslist)

casePlot_low.plot(dates, al_rateslist)



ax1.xaxis.set_major_locator(ticker.MultipleLocator(60))

casePlot_low.title("Percentage of State's Population Testing Positive Each Day (Least Dense)")

casePlot_low.legend(['South Dakota', 'North Dakota', 'Montana', 'Wyoming', 'Alaska'])

fig.savefig('lowdens_cases.png', dpi=150)

casePlot_low.show
fig, ax1 = plt.subplots(figsize = (15,6)) 

plt.style.use('ggplot')



x = ['District of Columbia', 'New Jersey' , 'Puerto Rico', 'Rhode Island', 'Massachusetts',

     'South Dakota', 'North Dakota', 'Montana', 'Wyoming', 'Alaska']



mean = [mr_dc, mr_nj, mr_pr, mr_ri, mr_ma, mr_sd, mr_nd, mr_mt, mr_wy, mr_al]



x_pos = [i for i, _ in enumerate(x)]



plt.bar(x_pos, mean, color=['green','green','green','green','green', 'blue', 'blue','blue','blue','blue'])

plt.xlabel("State")

plt.ylabel("% of Pop Infected")

plt.title("Avg Daily % of Population Returning Positive Tests")



plt.xticks(x_pos, x)



plt.show()