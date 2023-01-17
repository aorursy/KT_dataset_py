# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#Any results you write to the current directory are saved as output.
InjuryRecord = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")

PlayList = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")

PlayerTrackData = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")

PlayList['StadiumType'].unique()
exp2replace = ['Domed']

PlayList['StadiumType'] = PlayList['StadiumType'].replace(exp2replace, 'Dome')



exp2replace = ['Dome, closed', 'Domed, closed']

PlayList['StadiumType'] = PlayList['StadiumType'].replace(exp2replace, 'Closed Dome')



exp2replace = ['Cloudy']

PlayList['StadiumType'] = PlayList['StadiumType'].replace(exp2replace, '')



exp2replace = ['Indoors', 'Retractable Roof']

PlayList['StadiumType'] = PlayList['StadiumType'].replace(exp2replace, 'Indoor')



exp2replace = ['Oudoor','Ourdoor','Outddors', 'Outdoors','Outdor','Outside']

PlayList['StadiumType'] = PlayList['StadiumType'].replace(exp2replace, 'Outdoor')



exp2replace = ['Retr. Roof Closed','Retr. Roof-Closed', 'Indoor, Roof Closed']

PlayList['StadiumType'] = PlayList['StadiumType'].replace(exp2replace, 'Retr. Roof - Closed')



exp2replace = ['Retr. Roof-Open', 'Outdoor Retr Roof-Open', 'Indoor, Open Roof', 'Domed, Open', 'Domed, open', 'Open']

PlayList['StadiumType'] = PlayList['StadiumType'].replace(exp2replace, 'Retr. Roof - Open')



PlayList['StadiumType'].unique()
PlayList['Weather'].unique()
exp2replace = ['Indoor','Indoors']

PlayList['Weather'] = PlayList['Weather'].replace(exp2replace, )



exp2replace = ['Clear skies','Clear Skies']

PlayList['Weather'] = PlayList['Weather'].replace(exp2replace, 'Clear')



exp2replace = ['Overcast','Coudy', 'Clouidy', 'cloudy', 'Mostly Coudy', 'Mostly cloudy']

PlayList['Weather'] = PlayList['Weather'].replace(exp2replace, 'Cloudy')



exp2replace = ['Party']

PlayList['Weather'] = PlayList['Weather'].replace(exp2replace, 'Partly')



exp2replace = ['Partly clear','Partly Sunny', 'Party cloudy', 'Partly Clouidy', 'Partly cloudy']

PlayList['Weather'] = PlayList['Weather'].replace(exp2replace, 'Partly Cloudy')



exp2replace = ['Rain']

PlayList['Weather'] = PlayList['Weather'].replace(exp2replace, 'Rainy')



exp2replace = ['Clear and sunny', 'Sunny and clear', 'Sunny Skies']

PlayList['Weather'] = PlayList['Weather'].replace(exp2replace, 'Sunny')



exp2replace = ['Mostly Sunny Skies']

PlayList['Weather'] = PlayList['Weather'].replace(exp2replace, 'Mostly Sunny')



PlayList['Weather'].unique()

PlayList['Position'].unique()



exp2replace = ['Missing Data']

PlayList['Position'] = PlayList['Position'].replace(exp2replace, )

PlayList['PositionGroup'] = PlayList['PositionGroup'].replace(exp2replace, )



PlayList['PositionGroup'].unique()

import os

PlayList.to_csv('CleanPlaylist.csv')

#InjuryRecord.to_csv('InjuryRecord.csv')

#PlayerTrackData.to_csv('PlayerTrackData.csv')

files = os.listdir(os.curdir)

print (files)

import pandas as pd

edgedmaxspan = pd.read_csv("../input/nflnet/edgedmaxspan.csv")

#edgednfl = pd.read_csv("../input//nflnet/edgednfl.csv")

#edgednfull = pd.read_csv("../input/nflnet/edgednfull.csv")

#nodesmaxspan = pd.read_csv("../input/nflnet/nodesmaxspan.csv")

#nodesnfl = pd.read_csv("../input/nflnet/nodesnfl.csv")

#nodesnfull = pd.read_csv("../input/nflnet/nodesnfull.csv")
edgedmaxspan.head()
interpolated_variables = edgedmaxspan[['edget']].groupby('edget').count()

interpolated_variables