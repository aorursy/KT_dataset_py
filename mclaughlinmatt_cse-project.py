# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



DF = pd.read_csv("../input/shot_logs.csv")

# Any results you write to the current directory are saved as output.
DF
def recode(val):

    val2 = val[0:2]

    if val2[1] == ':':

        val2 = val2[0]

    num = int(val2)

    return(num)

def shot_clock_trans(row):

    if np.isnan(row['SHOT_CLOCK']):

        val2 = int(row['GAME_CLOCK'][-2:])

    else:

        val2 = row['SHOT_CLOCK']

    return(val2)





DF['SHOT_CLOCK'] = DF.apply(shot_clock_trans, axis=1)
print(DF.columns)

DF.sort_values('GAME_ID')



DFFixed=DF[['LOCATION','SHOT_NUMBER','PERIOD','GAME_CLOCK','SHOT_CLOCK'

 ,'DRIBBLES','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','player_name','FGM']]
DFFixed['SHOT_NUMBER']
GC_Bins = [-1,4,8,13]

GC_Cat = ['Early','Mid','Late']

SC_Bins = [-1,6,12,18,25]

SC_Cat = ['Early','Mid','Late','Emergency']

Drib_Bins =[-1,1,3,90] 

Drib_Cat = ['None','Quick','Long']

TT_Bins = [0,1,3,25]

TT_Cat = ['Short','Medium','Long']

SD_Bins = [0,4,15,22,24,100]

SD_Cat = ['PB','Close','Mid','TP','Long']

CDD_Bins = [0,1,3,5,100]

CDD_Cat = ['TG','Guarded','LG','Open']



#print(pd.datetime(DFFixed['GAME_CLOCK']))

DFFixed['GAME_CLOCK']=DFFixed['GAME_CLOCK'].apply(recode)

GCCategories = pd.cut(DFFixed['GAME_CLOCK'], GC_Bins, labels=GC_Cat)

DFFixed['GCCategories'] = pd.cut(DFFixed['GAME_CLOCK'], GC_Bins, labels=GC_Cat)

SCCategories = pd.cut(DFFixed['SHOT_CLOCK'], SC_Bins, labels=SC_Cat)

DFFixed['SCCategories'] = pd.cut(DFFixed['SHOT_CLOCK'], SC_Bins, labels=SC_Cat)

DribCategories = pd.cut(DFFixed['DRIBBLES'], Drib_Bins, labels=Drib_Cat)

DFFixed['DribCategories'] = pd.cut(DFFixed['DRIBBLES'], Drib_Bins, labels=Drib_Cat)

TTCategories = pd.cut(DFFixed['TOUCH_TIME'], TT_Bins, labels=TT_Cat)

DFFixed['TTCategories'] = pd.cut(DFFixed['TOUCH_TIME'], TT_Bins, labels=TT_Cat)

SDCategories = pd.cut(DFFixed['SHOT_DIST'], SD_Bins, labels=SD_Cat)

DFFixed['SDCategories'] = pd.cut(DFFixed['SHOT_DIST'], SD_Bins, labels=SD_Cat)

CDDCategories = pd.cut(DFFixed['CLOSE_DEF_DIST'], CDD_Bins, labels=CDD_Cat)

DFFixed['CDDCategories'] = pd.cut(DFFixed['CLOSE_DEF_DIST'], CDD_Bins, labels=CDD_Cat)

DFFixed.to_csv("ShotLogs.csv")
dfhist = DFFixed.loc[DFFixed['TOUCH_TIME'] >= 0]

dfhist['TOUCH_TIME'].hist(bins=24)

#dfhist = DFFixed.loc[DFFixed['CLOSE_DEF_DIST'] >= 0]

#print(dfhist['CLOSE_DEF_DIST'].describe())

dfhist['CLOSE_DEF_DIST'].hist(bins=24)