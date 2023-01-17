import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



dfResults = pd.read_csv('../input/ThursdayNight.csv')

dfResults = dfResults[dfResults['Name'] == 'Anthony'][['19-May','2-Jun','16-Jun','30-Jun','14-Jul','11-Aug','8-Sep','22-Sep','6-Oct']].transpose()

dfResults['AnthonyResults'] = pd.to_numeric(dfResults[27])
completedRaces = ['19-May','2-Jun','16-Jun','30-Jun','14-Jul','11-Aug','8-Sep']

z = np.polyfit(x=range(0,len(dfResults['AnthonyResults'][completedRaces])), y=pd.to_numeric(dfResults['AnthonyResults'][completedRaces]), deg=1)

p = np.poly1d(z)

dfResults['Trendline'] = p(range(0,len(dfResults['AnthonyResults'])))

dfResults[['AnthonyResults','Trendline']].plot(title="Anthony's results and projections for the 2016 season")