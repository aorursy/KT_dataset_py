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
#read_excel method from pandas 

#use sheet_name argument to get a specific sheet

#by default, gets the first sheet (number 0)

data = pd.read_excel('/kaggle/input/penalty-statistics-20192020/19-20XL_V2.xlsx')



data_pl = pd.read_excel('/kaggle/input/penalty-statistics-20192020/19-20XL_V2.xlsx', sheet_name = "PL 19-20")

print(data_pl)
but = data_pl.iloc[3,7]

print(but)
nb_pen_pl=95

nb_pen_scored_pl=0

for i in range(0,95):

    if data_pl.iloc[i,13]=='YES':

        nb_pen_scored_pl = nb_pen_scored_pl + 1



print('number of penaltys scored:',nb_pen_scored_pl)

perc_pen_scored_pl = nb_pen_scored_pl / nb_pen_pl

print('Percentage of success in penaltys:',perc_pen_scored_pl)
nb_pen_ltd_pl=0

nb_pen_dtw_pl=0

nb_pen_lnd_pl=0

nb_pen_wnd_pl=0



for i in range(0,95):

    if data_pl.iloc[i,9]=='YES':

        nb_pen_ltd_pl = nb_pen_ltd_pl + 1

for i in range(0,95):

    if data_pl.iloc[i,10]=='YES':

        nb_pen_dtw_pl = nb_pen_dtw_pl + 1

for i in range(0,95):

    if data_pl.iloc[i,11]=='YES':

        nb_pen_lnd_pl = nb_pen_lnd_pl + 1

for i in range(0,95):

    if data_pl.iloc[i,12]=='YES':

        nb_pen_wnd_pl = nb_pen_wnd_pl + 1

        

print('number of penaltys LTD:',nb_pen_ltd_pl)

perc_pen_ltd_pl = nb_pen_ltd_pl / nb_pen_pl

print('Percentage of LTD penaltys:',perc_pen_ltd_pl)



print('number of penaltys DTW:',nb_pen_dtw_pl)

perc_pen_dtw_pl = nb_pen_dtw_pl / nb_pen_pl

print('Percentage of DTW penaltys:',perc_pen_dtw_pl)



print('number of penaltys LND:',nb_pen_lnd_pl)

perc_pen_lnd_pl = nb_pen_lnd_pl / nb_pen_pl

print('Percentage of LND penaltys:',perc_pen_lnd_pl)



print('number of penaltys WND:',nb_pen_wnd_pl)

perc_pen_wnd_pl = nb_pen_wnd_pl / nb_pen_pl

print('Percentage of WND penaltys:',perc_pen_wnd_pl)
nbsc_pen_ltd_pl=0

nbsc_pen_dtw_pl=0

nbsc_pen_lnd_pl=0

nbsc_pen_wnd_pl=0



for i in range(0,95):

    if data_pl.iloc[i,9]=='YES':

        if data_pl.iloc[i,13] == 'YES':

            nbsc_pen_ltd_pl = nbsc_pen_ltd_pl + 1

print('nb of ltd pen scored:',nbsc_pen_ltd_pl)

perc_sc_pen_ltd_pl = nbsc_pen_ltd_pl / nb_pen_ltd_pl

print('Percentage of success in LTD penaltys:',perc_sc_pen_ltd_pl)



for i in range(0,95):

    if data_pl.iloc[i,10]=='YES':

        if data_pl.iloc[i,13] == 'YES':

            nbsc_pen_dtw_pl = nbsc_pen_dtw_pl + 1

print('nb of dtw pen scored:',nbsc_pen_dtw_pl)

perc_sc_pen_dtw_pl = nbsc_pen_dtw_pl / nb_pen_dtw_pl

print('Percentage of success in DTW penaltys:',perc_sc_pen_dtw_pl)



for i in range(0,95):

    if data_pl.iloc[i,11]=='YES':

        if data_pl.iloc[i,13] == 'YES':

            nbsc_pen_lnd_pl = nbsc_pen_lnd_pl + 1

print('nb of lnd pen scored:',nbsc_pen_lnd_pl)

perc_sc_pen_lnd_pl = nbsc_pen_lnd_pl / nb_pen_lnd_pl

print('Percentage of success in LND penaltys:',perc_sc_pen_lnd_pl)



for i in range(0,95):

    if data_pl.iloc[i,12]=='YES':

        if data_pl.iloc[i,13] == 'YES':

            nbsc_pen_wnd_pl = nbsc_pen_wnd_pl + 1

print('nb of wnd pen scored:',nbsc_pen_wnd_pl)

perc_sc_pen_wnd_pl = nbsc_pen_wnd_pl / nb_pen_wnd_pl

print('Percentage of success in WND penaltys:',perc_sc_pen_wnd_pl)
nb_d_pen_pl = nb_pen_ltd_pl + nb_pen_dtw_pl

nbsc_d_pen_pl = nbsc_pen_ltd_pl + nbsc_pen_dtw_pl

nb_nd_pen_pl = nb_pen_lnd_pl + nb_pen_wnd_pl

nbsc_nd_pen_pl = nbsc_pen_lnd_pl + nbsc_pen_wnd_pl



perc_sc_d_pen_pl = nbsc_d_pen_pl / nb_d_pen_pl

perc_sc_nd_pen_pl = nbsc_nd_pen_pl / nb_nd_pen_pl



print('Success in Decisive penalties:',perc_sc_d_pen_pl)

print('Success in non-Decisive penalties:',perc_sc_nd_pen_pl)
nb_pen_home_pl=0

nb_pen_away_pl=0



for i in range(0,95):

    if data_pl.iloc[i,6]=='Home':

        nb_pen_home_pl = nb_pen_home_pl + 1



for i in range(0,95):

    if data_pl.iloc[i,6]=='Away':

        nb_pen_away_pl = nb_pen_away_pl + 1



print('number of penaltys home:',nb_pen_home_pl)

perc_pen_home_pl = nb_pen_home_pl / nb_pen_pl

print('Percentage of Home penaltys:',perc_pen_home_pl)

print('number of penaltys away:',nb_pen_away_pl)

perc_pen_away_pl = nb_pen_away_pl / nb_pen_pl

print('Percentage of away penaltys:',perc_pen_away_pl)
nbsc_pen_home_pl=0

nbsc_pen_away_pl=0



for i in range(0,95):

    if data_pl.iloc[i,6]=='Home':

        if data_pl.iloc[i,13] == 'YES':

            nbsc_pen_home_pl = nbsc_pen_home_pl + 1

for i in range(0,95):

    if data_pl.iloc[i,6]=='Away':

        if data_pl.iloc[i,13] == 'YES':

            nbsc_pen_away_pl = nbsc_pen_away_pl + 1



print('number of penaltys home scored:',nbsc_pen_home_pl)

perc_sc_pen_home_pl = nbsc_pen_home_pl / nb_pen_home_pl

print('Percentage of Home penaltys scored:',perc_sc_pen_home_pl)

print('number of penaltys away scored:',nbsc_pen_away_pl)

perc_sc_pen_away_pl = nbsc_pen_away_pl / nb_pen_away_pl

print('Percentage of Away penaltys scored:',perc_sc_pen_away_pl)
nb_pen_st_pl=0

nb_pen_w_pl=0

nb_pen_om_pl=0

nb_pen_dm_pl=0

nb_pen_ld_pl=0

nb_pen_cb_pl=0

nb_pen_gk_pl=0



for i in range(0,95):

    if data_pl.iloc[i,15]=='ST':

        nb_pen_st_pl = nb_pen_st_pl + 1

    if data_pl.iloc[i,15]=='W':

        nb_pen_w_pl = nb_pen_w_pl + 1

    if data_pl.iloc[i,15]=='OM':

        nb_pen_om_pl = nb_pen_om_pl + 1

    if data_pl.iloc[i,15]=='DM':

        nb_pen_dm_pl = nb_pen_dm_pl + 1

    if data_pl.iloc[i,15]=='LD':

        nb_pen_ld_pl = nb_pen_ld_pl + 1

    if data_pl.iloc[i,15]=='CB':

        nb_pen_cb_pl = nb_pen_cb_pl + 1

    if data_pl.iloc[i,15]=='GK':

        nb_pen_gk_pl = nb_pen_gk_pl + 1



print('Number of pens shot by strikers:',nb_pen_st_pl)

print('Number of pens shot by wingers:',nb_pen_w_pl)

print('Number of pens shot by offensive midfielders:',nb_pen_om_pl)

print('Number of pens shot by defensive midfielders:',nb_pen_dm_pl)

print('Number of pens shot by lateral defenders:',nb_pen_ld_pl)

print('Number of pens shot by centerbacks:',nb_pen_cb_pl)

print('Number of pens shot by goalkeepers:',nb_pen_gk_pl)
nbsc_pen_st_pl=0

nbsc_pen_w_pl=0

nbsc_pen_om_pl=0

nbsc_pen_dm_pl=0

nbsc_pen_ld_pl=0

nbsc_pen_cb_pl=0

nbsc_pen_gk_pl=0



for i in range(0,95):

    if data_pl.iloc[i,15]=='ST':

        if data_pl.iloc[i,13] == 'YES':

            nbsc_pen_st_pl = nbsc_pen_st_pl + 1

    if data_pl.iloc[i,15]=='W':

        if data_pl.iloc[i,13] == 'YES':

            nbsc_pen_w_pl = nbsc_pen_w_pl + 1

    if data_pl.iloc[i,15]=='OM':

        if data_pl.iloc[i,13] == 'YES':

            nbsc_pen_om_pl = nbsc_pen_om_pl + 1

    if data_pl.iloc[i,15]=='DM':

        if data_pl.iloc[i,13] == 'YES':

            nbsc_pen_dm_pl = nbsc_pen_dm_pl + 1

    if data_pl.iloc[i,15]=='LD':

        if data_pl.iloc[i,13] == 'YES':

            nbsc_pen_ld_pl = nbsc_pen_ld_pl + 1

    if data_pl.iloc[i,15]=='CB':

        if data_pl.iloc[i,13] == 'YES':

            nbsc_pen_cb_pl = nbsc_pen_cb_pl + 1

    if data_pl.iloc[i,15]=='GK':

        if data_pl.iloc[i,13] == 'YES':

            nbsc_pen_gk_pl = nbsc_pen_gk_pl + 1





perc_sc_pen_st_pl = (nbsc_pen_st_pl / nb_pen_st_pl)*100

perc_sc_pen_w_pl = (nbsc_pen_w_pl / nb_pen_w_pl)*100

perc_sc_pen_om_pl = (nbsc_pen_om_pl / nb_pen_om_pl)*100

perc_sc_pen_dm_pl = (nbsc_pen_dm_pl / nb_pen_dm_pl)*100

perc_sc_pen_ld_pl = (nbsc_pen_ld_pl / nb_pen_ld_pl)*100

            

print('Number of pens scored by strikers:',nbsc_pen_st_pl,'(',perc_sc_pen_st_pl,'%)')

print('Number of pens scored by wingers:',nbsc_pen_w_pl,'(',perc_sc_pen_w_pl,'%)')

print('Number of pens scored by offensive midfielders:',nbsc_pen_om_pl,'(',perc_sc_pen_om_pl,'%)')

print('Number of pens scored by defensive midfielders:',nbsc_pen_dm_pl,'(',perc_sc_pen_dm_pl,'%)')

print('Number of pens scored by lateral defenders:',nbsc_pen_ld_pl,'(',perc_sc_pen_ld_pl,'%)')

print('Number of pens scored by centerbacks:',nbsc_pen_cb_pl,'(--%)')

print('Number of pens scored by goalkeepers:',nbsc_pen_gk_pl,'(--%)')
#read_excel method from pandas 

#use sheet_name argument to get a specific sheet

#by default, gets the first sheet (number 0)

data = pd.read_excel('/kaggle/input/penalty-statistics-20192020/19-20XL_V2.xlsx')



data_l1 = pd.read_excel('/kaggle/input/penalty-statistics-20192020/19-20XL_V2.xlsx', sheet_name = "ligue1 19-20")

print(data_l1)
nb_pen_l1=92

nb_pen_scored_l1=0

for i in range(0,92):

    if data_l1.iloc[i,13]=='YES':

        nb_pen_scored_l1 = nb_pen_scored_l1 + 1



print('number of penaltys scored:',nb_pen_scored_l1)

perc_pen_scored_l1 = nb_pen_scored_l1 / nb_pen_l1

print('Percentage of success in penaltys:',perc_pen_scored_l1)
nb_pen_ltd_l1=0

nb_pen_dtw_l1=0

nb_pen_lnd_l1=0

nb_pen_wnd_l1=0



for i in range(0,92):

    if data_l1.iloc[i,9]=='YES':

        nb_pen_ltd_l1 = nb_pen_ltd_l1 + 1

for i in range(0,92):

    if data_l1.iloc[i,10]=='YES':

        nb_pen_dtw_l1 = nb_pen_dtw_l1 + 1

for i in range(0,92):

    if data_l1.iloc[i,11]=='YES':

        nb_pen_lnd_l1 = nb_pen_lnd_l1 + 1

for i in range(0,92):

    if data_l1.iloc[i,12]=='YES':

        nb_pen_wnd_l1 = nb_pen_wnd_l1 + 1

        

print('number of penaltys LTD:',nb_pen_ltd_l1)

perc_pen_ltd_l1 = nb_pen_ltd_l1 / nb_pen_l1

print('Percentage of LTD penaltys:',perc_pen_ltd_l1)



print('number of penaltys DTW:',nb_pen_dtw_l1)

perc_pen_dtw_l1 = nb_pen_dtw_l1 / nb_pen_l1

print('Percentage of DTW penaltys:',perc_pen_dtw_l1)



print('number of penaltys LND:',nb_pen_lnd_l1)

perc_pen_lnd_l1 = nb_pen_lnd_l1 / nb_pen_l1

print('Percentage of LND penaltys:',perc_pen_lnd_l1)



print('number of penaltys WND:',nb_pen_wnd_l1)

perc_pen_wnd_l1 = nb_pen_wnd_l1 / nb_pen_l1

print('Percentage of WND penaltys:',perc_pen_wnd_l1)
nbsc_pen_ltd_l1=0

nbsc_pen_dtw_l1=0

nbsc_pen_lnd_l1=0

nbsc_pen_wnd_l1=0



for i in range(0,92):

    if data_l1.iloc[i,9]=='YES':

        if data_l1.iloc[i,13] == 'YES':

            nbsc_pen_ltd_l1 = nbsc_pen_ltd_l1 + 1

print('nb of ltd pen scored:',nbsc_pen_ltd_l1)

perc_sc_pen_ltd_l1 = nbsc_pen_ltd_l1 / nb_pen_ltd_l1

print('Percentage of success in LTD penaltys:',perc_sc_pen_ltd_l1)



for i in range(0,92):

    if data_l1.iloc[i,10]=='YES':

        if data_l1.iloc[i,13] == 'YES':

            nbsc_pen_dtw_l1 = nbsc_pen_dtw_l1 + 1

print('nb of dtw pen scored:',nbsc_pen_dtw_l1)

perc_sc_pen_dtw_l1 = nbsc_pen_dtw_l1 / nb_pen_dtw_l1

print('Percentage of success in DTW penaltys:',perc_sc_pen_dtw_l1)



for i in range(0,92):

    if data_l1.iloc[i,11]=='YES':

        if data_l1.iloc[i,13] == 'YES':

            nbsc_pen_lnd_l1 = nbsc_pen_lnd_l1 + 1

print('nb of lnd pen scored:',nbsc_pen_lnd_l1)

perc_sc_pen_lnd_l1 = nbsc_pen_lnd_l1 / nb_pen_lnd_l1

print('Percentage of success in LND penaltys:',perc_sc_pen_lnd_l1)



for i in range(0,92):

    if data_l1.iloc[i,12]=='YES':

        if data_l1.iloc[i,13] == 'YES':

            nbsc_pen_wnd_l1 = nbsc_pen_wnd_l1 + 1

print('nb of wnd pen scored:',nbsc_pen_wnd_l1)

perc_sc_pen_wnd_l1 = nbsc_pen_wnd_l1 / nb_pen_wnd_l1

print('Percentage of success in WND penaltys:',perc_sc_pen_wnd_l1)
nb_d_pen_l1 = nb_pen_ltd_l1 + nb_pen_dtw_l1

nbsc_d_pen_l1 = nbsc_pen_ltd_l1 + nbsc_pen_dtw_l1

nb_nd_pen_l1 = nb_pen_lnd_l1 + nb_pen_wnd_l1

nbsc_nd_pen_l1 = nbsc_pen_lnd_l1 + nbsc_pen_wnd_l1



perc_sc_d_pen_l1 = (nbsc_d_pen_l1 / nb_d_pen_l1)*100

perc_sc_nd_pen_l1 = (nbsc_nd_pen_l1 / nb_nd_pen_l1)*100



print('Success in Decisive penalties:',perc_sc_d_pen_l1)

print('Success in non-Decisive penalties:',perc_sc_nd_pen_l1)
nb_pen_home_l1=0

nb_pen_away_l1=0



for i in range(0,92):

    if data_l1.iloc[i,6]=='Home':

        nb_pen_home_l1 = nb_pen_home_l1 + 1



for i in range(0,92):

    if data_l1.iloc[i,6]=='Away':

        nb_pen_away_l1 = nb_pen_away_l1 + 1



print('number of penaltys home:',nb_pen_home_l1)

perc_pen_home_l1 = nb_pen_home_l1 / nb_pen_l1

print('Percentage of Home penaltys:',perc_pen_home_l1)

print('number of penaltys away:',nb_pen_away_l1)

perc_pen_away_l1 = nb_pen_away_l1 / nb_pen_l1

print('Percentage of away penaltys:',perc_pen_away_l1)



nbsc_pen_home_l1=0

nbsc_pen_away_l1=0



for i in range(0,92):

    if data_l1.iloc[i,6]=='Home':

        if data_l1.iloc[i,13] == 'YES':

            nbsc_pen_home_l1 = nbsc_pen_home_l1 + 1

for i in range(0,92):

    if data_l1.iloc[i,6]=='Away':

        if data_l1.iloc[i,13] == 'YES':

            nbsc_pen_away_l1 = nbsc_pen_away_l1 + 1



print('number of penaltys home scored:',nbsc_pen_home_l1)

perc_sc_pen_home_l1 = nbsc_pen_home_l1 / nb_pen_home_l1

print('Percentage of Home penaltys scored:',perc_sc_pen_home_l1)

print('number of penaltys away scored:',nbsc_pen_away_l1)

perc_sc_pen_away_l1 = nbsc_pen_away_l1 / nb_pen_away_l1

print('Percentage of Away penaltys scored:',perc_sc_pen_away_l1)