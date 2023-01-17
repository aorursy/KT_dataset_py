import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

df['Date'] = pd.to_datetime(df['Date'])



df['Province_State'] = '_'+df['Province_State']

df['Province_State'].fillna("",inplace=True)

df['Location'] = df['Country_Region']+df['Province_State']
locations = []

for loc,gdf in df.groupby('Location'):

    

    if gdf.ConfirmedCases.max()>200:

        locations.append(loc)

        

df = df[df.Location.isin(locations)]
def plotCases(df,loc):

    gdf = df.loc[df.Location==loc]

    plt.scatter(range(len(gdf.ConfirmedCases)),gdf.ConfirmedCases)



def plotDifference(df,loc):

    '''plots new cases on each day'''

    gdf = df.loc[df.Location==loc]

    y = list(gdf.ConfirmedCases)

    y = [y[i+1]-y[i] for i in range(len(y)-1)]

    

    plt.plot(range(len(y)),y)
plotCases(df,'Diamond Princess')
plotDifference(df,'Diamond Princess')
plotCases(df,'Belarus')
plotDifference(df,'Belarus')
def getSkewnessDict(df):

    

    skewDict = {}

    

    for loc,gdf in df.groupby('Location'):

        

        y = list(gdf['ConfirmedCases'])

        y = [y[i+1]-y[i]+1 for i in range(len(y)-1)]

        

        meanY = np.average(range(1,len(y)+1),weights=y)

        var = 0

        sumY = sum(y)

        for i,yd in enumerate(y,start=1):

            var += (yd*((i-meanY)**2))/sumY

        std = np.sqrt(var)

        skewness = sum((yd*(i-meanY)**3)/(sumY*std**3) for i,yd in enumerate(y,start=1))

        

        skewDict[loc] = skewness

        

    return skewDict
skDict = getSkewnessDict(df)

skDf = pd.DataFrame({"Location":list(sorted(skDict,key=skDict.get,reverse=True))

                     ,"Skewness":list(sorted(skDict.values(),reverse=True))})

skDf
(skDf.Skewness>0).sum()
plotCases(df,'Italy')
plotCases(df,'Spain')
plotCases(df,'Germany')