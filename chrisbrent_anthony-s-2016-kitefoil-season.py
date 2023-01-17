import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pprint import pprint



dfResults = pd.read_csv('../input/ThursdayNight.csv')



#dfResults = dfResults[dfResults['Name'] == 'Anthony'][['19-May','2-Jun','16-Jun','30-Jun','14-Jul','11-Aug','8-Sep','22-Sep','6-Oct']].transpose()

#pprint(dfResults)

#dfResults['AnthonyResults'] = pd.to_numeric(dfResults[27])





dfResults = dfResults[dfResults['Name'] == 'Chris'][['21-Apr','2-Jun','16-Jun','30-Jun','14-Jul','11-Aug','8-Sep','22-Sep','6-Oct']].transpose()

dfResults['ChrisResults'] = pd.to_numeric(dfResults[14])

pprint(dfResults)
completedRaces = ['21-Apr','2-Jun','16-Jun','30-Jun','14-Jul','11-Aug','8-Sep']

z = np.polyfit(x=range(0,len(dfResults['ChrisResults'][completedRaces])), y=pd.to_numeric(dfResults['ChrisResults'][completedRaces]), deg=1)

p = np.poly1d(z)

dfResults['Trendline'] = p(range(0,len(dfResults['ChrisResults'])))

dfResults[['ChrisResults','Trendline']].plot(title="Chris's results and projections for the 2016 season")