# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from scipy.fftpack import fft

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from dateutil.relativedelta import relativedelta

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.lines as mlines

import os





# Any results you write to the current directory are saved as output.

VanData = pd.read_csv("../input/vancouverdata/multiTimeline.csv")

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

prop = fm.FontProperties(fname='../input/robotolight/Roboto-Light.ttf')

prop_bold = fm.FontProperties(fname='../input/roboto/Roboto-Black.ttf')



colors = [[213/255,94/255,0], [86/255,180/255,233/255], [0,158/255,115/255,0.6],[0,0,0], [230/255,159/255,0], 

          [0,114/255,178/255]]



plt.style.use('fivethirtyeight')

VanData['Month']=pd.to_datetime(VanData['Month'], format='%Y-%m')

fte_graph = VanData.plot(x = 'Month', y = ['Vancouver'], color = colors, figsize = (12,8), legend = False)



fte_graph.tick_params(axis = 'both', which = 'major', labelsize = 18)

fte_graph.set_yticklabels(labels = [-10, '0  ', '20  ', '40  ', '60  ', '80  ', 'MAX'],fontproperties=prop, fontsize = 18,alpha = .7)

fte_graph.set_xticklabels(labels = ['2005', '07\'','09\'', '11\'', '13\'', '15\'', '17\'', '19\''],fontproperties=prop, fontsize = 18,alpha = .7)



fte_graph.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)

fte_graph.xaxis.label.set_visible(False)



# # The other signature bar

fte_graph.text(x = 395, y = -15,fontproperties=prop,

    s = '_________________________________________________________________________________________________________________________________________________________________________________________',

    color = 'grey', alpha = .7)



fte_graph.text(x = 395, y = -20,fontproperties=prop,

    s = ' © KAPastor                                                                                                                         Source: Google Trends   ',

    fontsize = 18, color = 'grey',  alpha = .7)





# Adding a title and a subtitle

fte_graph.text(x = 395, y = 117,fontproperties=prop_bold, s = "  US interest in Vancouver seems to be for vacation, Olympics or riots",

               fontsize = 26,  alpha = .75)

fte_graph.text(x = 395, y = 107,fontproperties=prop,

               s = '    US Google Trends search-index on Vancouver from 2004-Present shows periodic interest in June/July \n    or major events (Olympics or Riots)',

              fontsize = 16, alpha = .85)

fte_graph.text(x = 560, y = 0,fontproperties=prop,fontsize = 15, s = 'Vancouver Trends', color = colors[0], weight = 'bold', rotation = 0,

              backgroundcolor = '#f0f0f0')



l = mlines.Line2D([460,479], [70,90],linewidth = 0.5, color = 'grey', alpha = .7)

fte_graph.add_line(l)

fte_graph.text(x = 430, y = 70,fontproperties=prop,s = '2010 Winter Olympics', fontsize = 18, color = 'grey',  rotation = 0, alpha = .7, backgroundcolor = '#f0f0f0')



l = mlines.Line2D([500,520], [30,50],linewidth = 0.5, color = 'grey', alpha = .7)

fte_graph.add_line(l)

fte_graph.text(x = 510, y =50,fontproperties=prop,s = '2011 Vancouver Stanley Cup Riot*', fontsize = 18, color = 'grey',  rotation = 0, alpha = .7, backgroundcolor = '#f0f0f0')



fte_graph.text(x = 558, y = -13,fontproperties=prop,fontsize = 10, s = '* A proud moment in Canadian history', color = colors[0], weight = 'bold', rotation = 0,

              backgroundcolor = '#f0f0f0')



plt.show()

plt.savefig('niceplot.png')
import pandas as pd

import pandas as pd

multiTimeline = pd.read_csv("../input/multiTimeline.csv")