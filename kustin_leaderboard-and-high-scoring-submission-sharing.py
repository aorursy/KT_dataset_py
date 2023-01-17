# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



import math as math

%matplotlib inline

import seaborn as sns # charts

import matplotlib.pyplot as plt  #charts

import matplotlib.patches as patches

from matplotlib.gridspec import GridSpec

from datetime import datetime, timedelta

import matplotlib.dates as mdates

from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU

from matplotlib.dates import WeekdayLocator

from matplotlib.lines import Line2D



pbl = pd.read_csv("../input/swt2019pbljan4/santa-workshop-tour-2019-publicleaderboard-jan-4.csv", low_memory=False)

pbl = pbl.rename(columns={'SubmissionDate':'sDate'})

pbl['sDateH'] = pbl['sDate'].values.astype('<M8[h]')+pd.Timedelta(minutes=59,seconds=59)

pbl['sDateD'] = pbl['sDate'].values.astype('<M8[D]')

# teams first submission

pbl_tfsh = pbl.groupby(['TeamId'])['sDateH'].min().reset_index(name='sDateH')

pbl_tph = pbl_tfsh.sort_values(['sDateH']).groupby(['sDateH']).size().cumsum().reset_index(name='teams_cnt')

max_date = pbl['sDateH'].max()

def reindex_by_date(df):

    dates = pd.date_range(df.index.min(), max_date, freq='1H')

    return df.reindex(dates).ffill()

pbl_tph = pbl_tph.set_index('sDateH').apply(reindex_by_date).reset_index().rename(columns={'index':'sDateH'})

#pbl_tph.head()



#CALCULTING number of medals for each day based on number of teams



#          0-99 Teams   100-249    Teams 250-999      Teams 1000+ Teams

#  Bronze  Top 40%      Top 40%    Top 100            Top 10%

#  Silver  Top 20%      Top 20%    Top 50             Top 5%

#  Gold    Top 10%      Top 10     Top 10 + 0.2%*     Top 10 + 0.2%*



buckets = [99, 249, 999, np.inf]

pbl_tph['bucket_n'] = pd.cut(pbl_tph['teams_cnt'], [0]+buckets, labels=range(4))



medals_cnt = {

    'gold':

        {'abs':[0, 10, 10, 10], 

         'rel':[0.1, 0, 0.002, 0.002]},

    'silver':

        {'abs': [0, 0, 50, 0],

         'rel': [0.2, 0.2, 0, 0.05]},

    'bronze':

        {'abs':[0, 0, 100, 0],

         'rel':[0.4, 0.4, 0, 0.1]}}





for k,v in medals_cnt.items():

    key = f'{k}_medals_cnt'

    pbl_tph[key] = pbl_tph[['bucket_n', 'teams_cnt']].apply(

        lambda x: v['abs'][x['bucket_n']] + v['rel'][x['bucket_n']]*x['teams_cnt'], axis=1).astype('int')

#pbl_tph.head(5)



zh = pbl.loc[pbl.groupby(['TeamId','sDateH'])['Score'].idxmin()]

zh.head()

max_date = zh['sDateH'].max()



def reindex_by_date(df):

    dates = pd.date_range(df.index.min(), max_date, freq='1H')

    return df.reindex(dates).ffill()



zh = zh.set_index('sDateH').groupby('TeamId').apply(reindex_by_date).reset_index(0, drop=True)

#zh.head()



scores_eoh = zh.reset_index().rename(columns={'index':'sDateH'})

scores_eoh['rank'] = scores_eoh.sort_values(['Score','sDate']).groupby(['sDateH']).cumcount()+1

yy = pd.merge(scores_eoh[['Score','sDateH','rank']], pbl_tph, how='inner',on='sDateH')

pbl_cutoffs_h = yy[yy.filter(like='_medals_').eq(yy['rank'],axis=0).any(axis=1)].copy()

pbl_cutoffs_h['zone_x_ends'] = np.argmax(pbl_cutoffs_h.filter(like='medals').eq(pbl_cutoffs_h['rank'], axis=0).values, axis=1)

pbl_cutoffs_h['zone_x_ends'] = pbl_cutoffs_h['zone_x_ends'].map({0:'gold', 1:'silver', 2:'bronze'})
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



plt.rc('font', size=12) #controls default text sizes

plt.rc('axes', titlesize=14) #fontsize of the axes title

plt.rc('axes', labelsize=12) #fontsize of the x and y labels

plt.rc('xtick', labelsize=12) #fontsize of the tick labels

plt.rc('ytick', labelsize=12) #fontsize of the tick labels

plt.rc('legend', fontsize=12) #legend fontsize

plt.rc('figure', titlesize=14) #fontsize of the figure title

sns.set_style("whitegrid")



#data for plots

d= pbl_cutoffs_h[['sDateH','zone_x_ends','Score']].pivot(index = 'sDateH',columns='zone_x_ends', values='Score')







pcolors = {'light': 

              { 'bronze':'#E9B87A','silver':'#E9E9E9', 'gold':'#FFD44A'},

          'dark': 

              { 'bronze':'#8E5B3D','silver':'#838280', 'gold':'#B88121'},

          'annot':'#C75146',

          'higlight':'#FFBC42'}







gs_kw = dict(wspace=0.25, hspace=0.25, width_ratios=[1], height_ratios=[0.75,0.55])

fig, ax = plt.subplots(2,1,figsize=(12,16), gridspec_kw=gs_kw)





x_starts = [pd.to_datetime('2019-12-01 23:59:59'),

            pd.to_datetime('2019-12-26 15:59:59')]

ndays = [30,2.2]

xintervals = [72,6]

dfmt = [mdates.DateFormatter('%d-%b\n%a'),mdates.DateFormatter('%d-%b\n%I%p')]

ymaxs = [77000, 71500]

y_starts = [68700, 68700]

subtitle = [("Great progress was made by teams during the first week of the competition."+

             "\nThen things were calm for a while. What happened after Christmas?"),

            'Race of notebooks with scores in the bronze zone on the 27th of December']

key_events = {0:[], #72398.91,71261.10,70964.11

            1:[70964.11,69983.82,69880.40,70405.11]}

axvspans = {

    'dates': [(x_starts[0],x_starts[0]+timedelta(days=7)), 

               (pd.to_datetime('2019-12-25 23:59:59'), pd.to_datetime('2019-12-28 23:59:59'))],

    'color': ['#88C16E','#FFBC42']

}







for i in range(len(axvspans)):

    ax[0].axvspan(axvspans['dates'][i][0], axvspans['dates'][i][1], color=axvspans['color'][i], alpha = 0.08)

    



custom_legend_ls = list(pcolors['dark'].keys())

custom_legend_hs = []

for x in custom_legend_ls:

    custom_legend_hs.append(Line2D([0], [0], color=pcolors['dark'][x],marker='o', markersize=10, lw=0))





for i in range(0,len(x_starts)):

    #plot    

    sns.lineplot(data=d.loc[(d.index>x_starts[i])&(d.index<=x_starts[i]+timedelta(days=ndays[i]))][['bronze','silver','gold']], palette=pcolors['dark'] ,dashes = False, ax=ax[i])

    #plot decoration

    c_ax = ax[i] 

    dmin = x_starts[i]

    loc = mdates.HourLocator(interval=xintervals[i])

    c_ax.xaxis.set_major_locator(loc)

    c_ax.xaxis.set_major_formatter(dfmt[i])

    c_ax.tick_params(labelsize=10)

    c_ax.set_xlabel(None)

    c_ax.set_ylim(y_starts[i], ymaxs[i])

    c_ax.set_title(subtitle[i])

    c_ax.legend(custom_legend_hs, custom_legend_ls, ncol=1)



    #plot annotation

    pn = key_events[i]

    if len(pn)>0:

        annot_points = pbl.loc[pbl['Score'].isin(pn)].groupby('Score')['sDateH'].min().sort_index(ascending=False).to_dict()

        for score, tstmp in annot_points.items():  

            #score of the last entry in the bronze zone prior sharing

            bronze_score_ts = d.loc[tstmp+timedelta(hours=-1),'bronze']

            #mark sharing time

            c_ax.vlines(tstmp+timedelta(hours=-1), y_starts[i],bronze_score_ts, colors='grey', alpha=1, linestyles='dotted',lw=2)

            #add a score of the shared notebook

            c_ax.scatter(tstmp+timedelta(hours=-1),score, color=pcolors['annot'], marker='^')

            c_ax.annotate(str(int(score))+"\nnotebook\npublished",

                        xy=(tstmp+timedelta(hours=-1),score),

                        xytext=(0,-10), textcoords='offset points',

                        va="top", ha="center",

                        color=pcolors['annot'],

                        bbox=dict(boxstyle='round', fc="w", ec=pcolors['annot'])   

                        )

            #add number of uploads

            uploads = pbl.loc[(pbl['Score']==score)&(pbl['sDateH']<=annot_points[score]+timedelta(hours=24))]['TeamId'].count()

            c_ax.annotate(str(uploads)+" uploads*",

                          xy=(tstmp+timedelta(hours=-1), score),

                          xytext=(0,-62), textcoords='offset points',

                          va="top", ha="center", color='grey', fontsize=10,

                          bbox=dict(boxstyle='round', fc="w",ec="w")   

                         )

            #current bronze score

            c_ax.annotate(str(int(bronze_score_ts)),

                        xy=(tstmp+timedelta(hours=-1),bronze_score_ts),

                        xytext=(0,0), textcoords='offset points',

                        va="bottom", ha="center",

                        color="w",

                        bbox=dict(boxstyle='round4', fc=pcolors['dark']['bronze'], ec=pcolors['dark']['bronze'])   

                        )

            

#add milestones on the first plot:

milestones = [pd.to_datetime('2019-12-01 23:59:59'),

              pd.to_datetime('2019-12-10 23:59:59'),

              pd.to_datetime('2019-12-20 23:59:59'),

              pd.to_datetime('2019-12-30 23:59:59')]



for i in milestones:

    for j in ['gold','silver','bronze']:

        cur_score = d.loc[i,j]

        ax[0].annotate(str(int(cur_score)),

                      xy=(i,cur_score),

                      xytext=(0,0), textcoords='offset points',

                      va="center", ha="center",

                      color="w",

                      bbox=dict(boxstyle='round4', fc=pcolors['dark'][j], ec=pcolors['dark'][j]))



ax[1].annotate("*submissions with the same exact score made within the next 24 hours",

              xy=(0, 0),xycoords='axes fraction', 

              xytext=(0,-50), textcoords='offset points',

              va="top", ha="left", color='grey',

              bbox=dict(boxstyle='round', fc="w", ec='w'))



sns.despine(left=True, bottom=True)

plt.show()