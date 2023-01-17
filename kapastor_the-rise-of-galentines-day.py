

import numpy as np # linear algebra

import datetime

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from dateutil.relativedelta import relativedelta

import matplotlib.lines as mlines

import matplotlib.dates as mdates



import os

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

prop = fm.FontProperties(fname='../input/googlefonts/Lacquer/Lacquer-Regular.ttf')

prop = fm.FontProperties(fname='../input/googlefonts/Lacquer/Lacquer-Regular.ttf')

prop = fm.FontProperties(fname='../input/googlefonts/Montserrat_Subrayada/MontserratSubrayada-Regular.ttf')

prop = fm.FontProperties(fname='../input/googlefonts/Nunito/Nunito-Regular.ttf')





prop_bold = fm.FontProperties(fname='../input/googlefonts/Nunito/Nunito-SemiBold.ttf')

colors = [[255/255, 175/255, 185/255]]

plt.style.use('fivethirtyeight')

years = mdates.YearLocator()   # every year

years_fmt = mdates.DateFormatter('%Y')



# Add a spline and clean up the data

GD = pd.read_csv("../input/galentinesdaygoogletrends/GalentinesDay.csv")

GD['Month']=pd.to_datetime(GD['Month'], format='%b-%y')

fig, GD_ax = plt.subplots()

GD_ax.set_facecolor("white")

fig.patch.set_facecolor("white")

GD_ax = GD.plot(ax=GD_ax,x = 'Month', y = ['Interest'], color = colors, figsize = (12,9),linewidth = 10, legend = False)

GD_ax.tick_params(axis = 'both', which = 'major', labelsize = 25)

GD_ax.set_yticklabels(labels = [-10, '0  ', '20  ', '40  ', '60  ', '80  ', 'MAX'],fontproperties=prop, fontsize = 25,alpha = .7)



GD_ax.set_xticklabels(labels = ['2011','2012','2013','2014','2015','2016','2017','2018','2019','2020'],fontproperties=prop, fontsize = 18,alpha = .7)



GD_ax.xaxis.label.set_visible(False)



GD_ax.text(x = 475, y = -15,fontproperties=prop,s = '______________________________________________________________________________________________________________________________________________________________________________________________',

    color = 'grey', alpha = .7)

GD_ax.text(x = 475, y = -20,fontproperties=prop,s = ' www.kaggle.com/kapastor                                                                                            Source: Google Trends   ',

    fontsize = 18, color = 'grey',  alpha = .7)



GD_ax.grid(False)



# Adding a title and a subtitle

# GD_ax.text(x = 475, y = 117,fontproperties=prop_bold, s = "  The Rise of Galentine's Day",

#                fontsize = 26,  alpha = .75)

# GD_ax.text(x = 475, y = 107,fontproperties=prop,

#                s = '     Ever since the episode of Parks and Rec Galentines day has been on the rise',

#               fontsize = 16, alpha = .85)

# GD_ax.text(x = 560, y = 0,fontproperties=prop,fontsize = 15, s = 'Vancouver Trends', color = colors[0], weight = 'bold', rotation = 0,

#               backgroundcolor = '#f0f0f0')



plt.show()

# plt.savefig('niceplot.png')








