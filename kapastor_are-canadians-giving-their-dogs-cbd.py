# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from scipy.fftpack import fft

import numpy as np # linear algebra

import datetime

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from dateutil.relativedelta import relativedelta

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.lines as mlines

import os

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

prop = fm.FontProperties(fname='../input/robotolight/Roboto-Light.ttf')

prop_bold = fm.FontProperties(fname='../input/roboto/Roboto-Black.ttf')

colors = [[93/255, 218/255, 161/255], [86/255,180/255,233/255], [0,158/255,115/255,0.6],[0,0,0,0.8], [230/255,159/255,0], 

          [0,114/255,178/255]]

plt.style.use('fivethirtyeight')

from scipy.interpolate import make_interp_spline, BSpline

epoch = datetime.datetime.utcfromtimestamp(0)



# Add a spline and clean up the data

CBD = pd.read_csv("../input/cbddogs/multiTimeline (4).csv")

CBD['Week']=pd.to_datetime(CBD['Week'], format='%Y-%m-%d')

CBD['Week'] = CBD['Week'].apply(lambda x: (x- epoch).total_seconds())





xnew = np.linspace(CBD['Week'].min(), CBD['Week'].max(), 100000) 

spl = make_interp_spline(CBD['Week'], CBD['CBD'],k=2)  # type: BSpline

power_smooth = spl(xnew)



z = np.polyfit(CBD['Week'], CBD['CBD'], 5)

p = np.poly1d(z)



fit = p(xnew)

xnew = [datetime.datetime.fromtimestamp(d) for d in xnew]



CBD['Week'] = [datetime.datetime.fromtimestamp(d) for d in CBD['Week']]



data = {'Week':xnew, 'CBD':fit} 

df2 = pd.DataFrame(data)



fig, cbd_ax = plt.subplots()

cbd_ax.set_facecolor("white")



fig.patch.set_facecolor("white")



cbd_ax = df2.plot(ax=cbd_ax,x = 'Week', y = ['CBD'], color = [colors[3]], figsize = (12,10), legend = False,linewidth=1)

l = mlines.Line2D([datetime.datetime(2017,4,13),datetime.datetime(2017,4,13)], [0,100],linewidth = 1, color = 'grey', alpha = .9,linestyle='--')

cbd_ax.add_line(l)

l = mlines.Line2D([datetime.datetime(2018,3,22),datetime.datetime(2018,3,22)], [0,100],linewidth = 1.5, color = 'grey', alpha = .9,linestyle='--')

cbd_ax.add_line(l)



l = mlines.Line2D([datetime.datetime(2018,10,17),datetime.datetime(2018,10,17)], [0,100],linewidth = 2, color = 'grey', alpha = .9)

cbd_ax.add_line(l)



cbd_ax = CBD.plot(ax=cbd_ax,x = 'Week', y = ['CBD'], color = [colors[2]], linewidth=0,linestyle='--', marker='o',figsize = (12,10), legend = False)

cbd_fig = cbd_ax.get_figure()

data = {'Week':xnew, 'CBD':fit} 

df2 = pd.DataFrame(data)

cbd_ax = df2.plot(ax=cbd_ax,x = 'Week', y = ['CBD'], color = [colors[3]], figsize = (12,10), legend = False,linewidth=3)











cbd_ax.text(x = datetime.datetime(2015,1,1), y = 115,fontproperties=prop_bold, s = "Canadian Interest in CBD for Dogs", fontsize = 26,  alpha = .75)

cbd_ax.text(x = datetime.datetime(2015,1,1), y = 109,fontproperties=prop,

               s = 'Animals can get anxious just like humans and it looks like Canadians are exploring CBD for dogs',fontsize = 16, alpha = .85)





cbd_ax.tick_params(axis = 'both', which = 'major', labelsize = 18)

cbd_ax.set_yticklabels(labels = [-10, '0  ', '20  ', '40  ', '60  ', '80  ', 'MAX'],fontproperties=prop, fontsize = 14,alpha = .7)

cbd_ax.set_xticklabels(labels = ['2015-07','2016-01','2016-07','2017-01','2017-07','2018-01','2018-07','2019-01','2019-07','2020-01'],fontproperties=prop, rotation=15,fontsize = 14,alpha = .7)



cbd_ax.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .3)

cbd_ax.xaxis.label.set_visible(False)

cbd_ax.text(x = datetime.datetime(2014,10,1), y = 70,fontproperties=prop,s = 'Google Trends Index', fontsize = 14, color = 'grey',  rotation = 90, alpha = .9, backgroundcolor = 'white')





cbd_ax.text(x = datetime.datetime(2018,4,5), y = 5,fontproperties=prop,fontsize = 15, s = 'Google Trends: "CBD for Dogs" (Canada)', color = 'darkgreen', weight = 'bold', rotation = 0,

              backgroundcolor = 'white')







l = mlines.Line2D([datetime.datetime(2016,6,1),datetime.datetime(2017,4,1)], [70,90],linewidth = 1, color = 'grey', alpha = .9)

cbd_ax.add_line(l)

cbd_ax.text(x = datetime.datetime(2015,6,1), y = 70,fontproperties=prop,s = 'Bill C-45 Introduced To Parliament', fontsize = 18, color = 'grey',  rotation = 0, alpha = .9, backgroundcolor = 'white')



l = mlines.Line2D([datetime.datetime(2017,1,1),datetime.datetime(2018,3,10)], [50,60],linewidth = 1, color = 'grey', alpha = .9)

cbd_ax.add_line(l)

cbd_ax.text(x = datetime.datetime(2017,1,1), y = 50,fontproperties=prop,s = 'Bill C-45 Approval', fontsize = 18, color = 'grey',  rotation = 0, alpha = .9, backgroundcolor = 'white')



l = mlines.Line2D([datetime.datetime(2019,5,17),datetime.datetime(2018,10,30)], [23,30],linewidth = 1, color = 'grey', alpha = .9)

cbd_ax.add_line(l)

cbd_ax.text(x = datetime.datetime(2019,5,17), y = 20,fontproperties=prop,s = 'Legalized', fontsize = 18, color = 'grey',  rotation = 0, alpha = .9, backgroundcolor = 'white')





cbd_ax.text(x = datetime.datetime(2018,4,5), y = 12,fontproperties=prop,fontsize = 10, s = '* Cannabidiol (CBD) ', color = 'darkgreen', weight = 'bold', rotation = 0,

              backgroundcolor = 'white')



# # The other signature bar

cbd_ax.text(x = datetime.datetime(2015,1,5), y = -17,fontproperties=prop,

    s = '______________________________________________________________________________________________________________________________________________________________________________',

    color = 'grey', alpha = .7)



cbd_ax.text(x = datetime.datetime(2015,1,5), y = -22,fontproperties=prop,

    s = ' © KAPastor                                                                                                                                                                Source: Google Trends   ',

    fontsize = 14, color = 'grey',  alpha = .7)









plt.show()

plt.savefig('niceplot.png')








