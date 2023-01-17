# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np 
import pandas as pd
season = pd.read_csv('../input/season78.csv',encoding ='utf-8')
season.tail()
print("Zaza Pachulia's good season")
zazagoodseason=season[(season.Player=="Zaza Pachulia") & (season.Season>=2005) &(season.WS>=4.0)]
print(zazagoodseason) #İnterestingly best Win-Shares his low usage Warriors year.
winshares=season["WS"]
print("Max Winshare in Nba history is ")
maxwinshares=max(winshares) 
bestwinshareperson=season[season.WS>21]
print(bestwinshareperson) #Michael JORDAN. His 1988 season is historic.
print("Average Win Share is ")
meanwinshares=winshares.mean()
print(meanwinshares)
print("Bad players season  in NBA history")
minplayer=season[season.WS<=-1.5]["Player"]
print(minplayer)
#and years of bad seasons
minyears=season[season.WS<=-1.5]["Season"] #Adam Morrision's best season is also very bad.
print(minyears)
print("Min win share player in history of NBA")
minwinshares=min(winshares)
minplayer=season[season.WS<=minwinshares] #Emmanuel Mudiay is worst season performance of history
print(minplayer)
print("Better than mean how many season")
betterthanmean=season[season.WS>meanwinshares].count()
print(betterthanmean) #5956 is very good number
standardeviation=winshares.std()
hedo=season[season.Player=="Hedo Turkoglu"]
hedosmaxWS=max(hedo["WS"])
print(standardeviation)
print(hedosmaxWS)
print("How many players-season Better Win Share than HEDO's best")
howManyBetterHedo=season[season.WS>hedosmaxWS].count()
print(howManyBetterHedo) 
season[season.Player=="Jamal Crawford"] #Jamal is very good player but his some season is too bad








