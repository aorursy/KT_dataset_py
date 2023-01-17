# *This is an introductory plot*



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data=pd.read_csv("../input/ghcn-m-v1.csv")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(data[(data['year']==1880) & (data['lat']=='30-35N')]['lon_130_135E'])

plt.ylabel('Temperature Anomalies * 100')

plt.xlabel('Year 1880 Month wise')
# Replace -9999 with 0

data=data.replace([-9999],[0])



longColumns = list(data.columns.values)

longColumns.remove('month')

longColumns.remove('lat')

longColumns.remove('year')



def getLat(x):

    pos=x.find('-')

    return(x[0:pos])

    

def getLon(x):

    temp=x[x.find('_')+1:x.find('_')+5]

    temp=temp[0:temp.find('_')]

    return(temp)



def applyFunction(data,func):

    if func=='mean':

        results=data.groupby(['lat'])[longColumns].mean()

    if func=='std':    

        results=data.groupby(['lat'])[longColumns].std()

        

    resultDF=pd.DataFrame()

    for index,row in results.iterrows():

        for index1,row1 in row.iteritems():

            resultDF=resultDF.append(pd.DataFrame([[index,index1,row1]], columns=['lat','lon','val']))



    # Cleaning the LAT LONG Numbers

    resultDF.head(5)

    #resultDF.ix[1:1,'lat']

    

    resultDF['newLAT'] = resultDF['lat'].apply(getLat)

    resultDF['newLON'] = resultDF['lon'].apply(getLon)

    resultDF['newLAT'] = resultDF['newLAT'].apply(lambda x : float(x))

    resultDF['newLON'] = resultDF['newLON'].apply(lambda x : float(x))

    resultDF['LATDir'] = resultDF['lat'].apply(lambda x: 1 if x.endswith('N') else -1)

    resultDF['LONDir'] = resultDF['lon'].apply(lambda x: 1 if x.endswith('E') else -1)

    resultDF['newLAT'] = resultDF.apply(lambda x : x['newLAT'] * x['LATDir'],axis=1)

    resultDF['newLON'] = resultDF.apply(lambda x : x['newLON'] * x['LONDir'],axis=1)

    

    return resultDF

    
## MEAN

resultDF=applyFunction(data,'mean')



# Simple Plot highlighting areas with the maximum deviation

from mpl_toolkits.basemap import Basemap

m = Basemap(projection='hammer',lon_0=80,lat_0=0)

x, y = m(list(resultDF['newLON']),list(resultDF['newLAT']))



# From the hist diagram we can see that the deviation values are quite high in some cases

resultDF['val'].hist()



m.drawmapboundary(fill_color='#99ffff')

m.fillcontinents(color='#cc9966',lake_color='#99ffff',alpha=0.7)



# We will be separating them out based on the values and the color them accordingly

tempDF=resultDF[(resultDF['val'] < 20) & (resultDF['val'] > -20)][['newLAT','newLON','val']]

x, y = m(list(tempDF['newLON']),list(tempDF['newLAT']))

m.scatter(x,y,s=5,marker='o',color='b',alpha=0.7,label='LOW')



tempDF=resultDF[((resultDF['val'] < 50) & (resultDF['val'] > 20)) | ((resultDF['val'] > -50) & (resultDF['val'] < -20))  ][['newLAT','newLON','val']]

x, y = m(list(tempDF['newLON']),list(tempDF['newLAT']))

m.scatter(x,y,s=5,marker='o',color='g',alpha=0.7,label='AVERAGE')



tempDF=resultDF[((resultDF['val'] < 100) & (resultDF['val'] > 50)) | ((resultDF['val'] > -100) & (resultDF['val'] < -50))  ][['newLAT','newLON','val']]

x, y = m(list(tempDF['newLON']),list(tempDF['newLAT']))

m.scatter(x,y,s=5,marker='o',color='r',alpha=0.7,label='HIGH')



plt.title('Distribution of Mean over the years')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
## STANDARD DEVIATION

resultDF=applyFunction(data,'std')



# Simple Plot highlighting areas with the maximum deviation

from mpl_toolkits.basemap import Basemap

m = Basemap(projection='hammer',lon_0=80,lat_0=0)

x, y = m(list(resultDF['newLON']),list(resultDF['newLAT']))



# From the hist diagram we can see that the deviation values are quite high in some cases

resultDF['val'].hist()



m.drawmapboundary(fill_color='#99ffff')

m.fillcontinents(color='#cc9966',lake_color='#99ffff',alpha=0.7)



# We will be separating them out based on the values and the color them accordingly

tempDF=resultDF[(resultDF['val'] < 50) & (resultDF['val'] > -50)][['newLAT','newLON','val']]

x, y = m(list(tempDF['newLON']),list(tempDF['newLAT']))

m.scatter(x,y,s=5,marker='o',color='b',alpha=0.7,label='LOW')



tempDF=resultDF[((resultDF['val'] < 100) & (resultDF['val'] > 50))][['newLAT','newLON','val']]

x, y = m(list(tempDF['newLON']),list(tempDF['newLAT']))

m.scatter(x,y,s=5,marker='o',color='g',alpha=0.7,label='AVERAGE')



tempDF=resultDF[((resultDF['val'] < 200) & (resultDF['val'] > 100)) ][['newLAT','newLON','val']]

x, y = m(list(tempDF['newLON']),list(tempDF['newLAT']))

m.scatter(x,y,s=5,marker='o',color='r',alpha=0.7,label='HIGH')



plt.title('Distribution of Standard Deviation over the years')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()



# We can see that the areas with high mean are not necessarily the areas with high standard deviation

# Let us dig further