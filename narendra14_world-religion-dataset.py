# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import os

import plotly.plotly as py

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

from pylab import figure, axes, pie, title, show

import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.max_columns', 100)



Religion = pd.read_csv('../input/regional.csv')

religionAll = Religion[['region','year','christianity_all','judaism_all','islam_all','buddhism_all','zoroastrianism_all','hinduism_all','sikhism_all','confucianism_all','noreligion_all']].copy() 

religionByAfrica = religionAll[religionAll['region']=="Africa"]



religionByAfrica.reset_index()

colors = ['b', 'g', 'r', 'c', 'm','y', 'k', 'w', '#FF0033', '#FF8833','#FF0088']

ax = religionByAfrica[['christianity_all','judaism_all','islam_all','buddhism_all','hinduism_all','sikhism_all','confucianism_all','noreligion_all']].plot(kind='bar', title ="African Population", color=colors)

ax.set_xlabel("Year",fontsize=12)

ax.set_ylabel("Population", fontsize=12)

ax.set_xticklabels(religionByAfrica['year'])





religionByAsia = religionAll[religionAll['region']=="Asia"]



religionByAsia.reset_index()



ax = religionByAsia[['christianity_all','judaism_all','islam_all','buddhism_all','hinduism_all','sikhism_all','confucianism_all','noreligion_all']].plot(kind='bar', title ="Asian Population", color=colors)

ax.set_xlabel("Year",fontsize=12)

ax.set_ylabel("Population", fontsize=12)

ax.set_xticklabels(religionByAfrica['year'])





religionByEurope = religionAll[religionAll['region']=="Europe"]



religionByEurope.reset_index()



ax = religionByEurope[['christianity_all','judaism_all','islam_all','buddhism_all','hinduism_all','sikhism_all','confucianism_all','noreligion_all']].plot(kind='bar', title ="European Population", color=colors)

ax.set_xlabel("Year",fontsize=12)

ax.set_ylabel("Population", fontsize=12)

ax.set_xticklabels(religionByEurope['year'])





religionByMidEast = religionAll[religionAll['region']=="Mideast"]



religionByMidEast.reset_index()



ax = religionByMidEast[['christianity_all','judaism_all','islam_all','buddhism_all','hinduism_all','sikhism_all','confucianism_all','noreligion_all']].plot(kind='bar', title ="Middle-East Population", color=colors)

ax.set_xlabel("Year",fontsize=12)

ax.set_ylabel("Population", fontsize=12)

ax.set_xticklabels(religionByMidEast['year'])

plt.show()

# Reading from National.csv



Religion = pd.read_csv('../input/national.csv')

religionAll = Religion[['state','year','code','christianity_all','judaism_all','islam_all','buddhism_all','zoroastrianism_all','hinduism_all','sikhism_all','confucianism_all','noreligion_all']].copy() 



#religionAll['state'].drop_duplicates()



religionByCountry_India = religionAll[religionAll['state']=='India']



religionByCountry_India.reset_index()



ax = religionByCountry_India[['christianity_all','judaism_all','islam_all','buddhism_all','hinduism_all','sikhism_all','confucianism_all','noreligion_all']].plot(kind='bar', title ="India Population", color=colors)

ax.set_xlabel("Year",fontsize=12)

ax.set_ylabel("Population", fontsize=12)

ax.set_xticklabels(religionByCountry_India['year'])



religionByCountry_Pak = religionAll[religionAll['state']=='Pakistan']



religionByCountry_Pak.reset_index()



ax = religionByCountry_Pak[['christianity_all','judaism_all','islam_all','buddhism_all','hinduism_all','sikhism_all','confucianism_all','noreligion_all']].plot(kind='bar', title ="Pakistan Population", color=colors)

ax.set_xlabel("Year",fontsize=12)

ax.set_ylabel("Population", fontsize=12)

ax.set_xticklabels(religionByCountry_Pak['year'])



religionByCountry_Saudi = religionAll[religionAll['state']=='Saudi Arabia']

religionByCountry_Saudi.reset_index()



ax = religionByCountry_Saudi[['christianity_all','judaism_all','islam_all','buddhism_all','hinduism_all','sikhism_all','confucianism_all','noreligion_all']].plot(kind='bar', title ="Saudi Population", color=colors)

ax.set_xlabel("Year",fontsize=12)

ax.set_ylabel("Population", fontsize=12)

ax.set_xticklabels(religionByCountry_Saudi['year'])



religionByCountry_China = religionAll[religionAll['state']=='China']

religionByCountry_China.reset_index()



ax = religionByCountry_China[['christianity_all','judaism_all','islam_all','buddhism_all','hinduism_all','sikhism_all','confucianism_all','noreligion_all']].plot(kind='bar', title ="China Population", color=colors)

ax.set_xlabel("Year",fontsize=12)

ax.set_ylabel("Population", fontsize=12)

ax.set_xticklabels(religionByCountry_China['year'])



religionByCountry_Russia = religionAll[religionAll['state']=='Russia']

religionByCountry_Russia.reset_index()



ax = religionByCountry_Russia[['christianity_all','judaism_all','islam_all','buddhism_all','hinduism_all','sikhism_all','confucianism_all','noreligion_all']].plot(kind='bar', title ="Russia Population", color=colors)

ax.set_xlabel("Year",fontsize=12)

ax.set_ylabel("Population", fontsize=12)

ax.set_xticklabels(religionByCountry_Russia['year'])

plt.show()