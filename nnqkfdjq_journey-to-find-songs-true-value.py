import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime 
spo = pd.read_csv('../input/spotifys-worldwide-daily-song-ranking/data.csv')
releasedate = pd.read_csv('../input/spotifys-daily-song-ranking-music-released-date/spotify_music_released_date.csv', header = None)
releasedate = releasedate.dropna()
def fam(x):
    return datetime.strptime(x[:10] , "%Y-%m-%d") 
mj = releasedate[1].map(fam)
releasedate.iloc[:,1] = mj
releasedate.head(20)
releasedate = releasedate[releasedate[1] >= datetime(2017,1,1)]
releasedate = releasedate[releasedate[1] <= datetime(2017,4,1)] 
tglist = list(releasedate[0])                                       
fg = lambda x : True if x in tglist else False
newspo = spo['Track Name'].map(fg) 
spo[newspo].head(20)
finaldf = pd.DataFrame(index = tglist, columns = ['value','streaminglevel', 'startdate', 'entlen' ,'morethan70len', 'maxlen'])
for iname in tglist:
    testRickey = spo[spo['Track Name'] ==iname].groupby('Date').sum()
    testRickey = testRickey.reset_index()
    susul2 = testRickey.loc[:,['Date','Streams']]
    susul2['Ra'] = range(len(susul2))
    ym = np.array(susul2['Streams'])
    xm = np.array(susul2['Ra'])
    zm = np.polyfit(xm, ym, 30)                                                   
    fm = np.poly1d(zm)                     
    x_Me = np.linspace(xm[0], xm[-1], len(testRickey))
    y_Me = fm(x_Me)                         
    newdict = { 'x' : x_Me, 'y' : y_Me}                 
    newdictdf = pd.DataFrame(newdict)  
    paramss = 0.7
    L = newdictdf['y'] >= (newdictdf['y'].max()*paramss)     
    newdictdf['TF'] = L                                         
    finaldf.loc[iname,'value'] = L.sum()/len(testRickey)
    finaldf.loc[iname,'streaminglevel']= y_Me.max()/100000
    finaldf.loc[iname,'startdate'] = susul2.iloc[0,0]
    finaldf.loc[iname,'entlen'] =  len(testRickey)
    finaldf.loc[iname,'morethan70len'] =  L.sum()
    finaldf.loc[iname,'maxlen'] =  371   
finaldf.head(10)
finaldfsort = finaldf[(finaldf['streaminglevel'] > 2)]
New_method_list = finaldfsort.sort_values(by = ['morethan70len','value'], ascending = False)
New_method_list.head(10)
ordinary_method_list = spo[newspo].groupby('Track Name').sum().sort_values(by = 'Streams', ascending = False)
ordinary_method_list.head(10)
fig, ax = plt.subplots(10, 2, figsize=(13,70))
for num, iname in enumerate(list(New_method_list.head(10).index)):
    testRickey = spo[spo['Track Name'] ==iname].groupby('Date').sum()
    testRickey = testRickey.reset_index()
    susul2 = testRickey.loc[:,['Date','Streams']]
    susul2['Ra'] = range(len(susul2))
    ym = np.array(susul2['Streams'])
    xm = np.array(susul2['Ra'])
    zm = np.polyfit(xm, ym, 30)  
    fm = np.poly1d(zm)    
    x_Me = np.linspace(xm[0], xm[-1], len(testRickey))
    y_Me = fm(x_Me)
    newdict = { 'x' : x_Me, 'y' : y_Me}                 
    newdictdf = pd.DataFrame(newdict) 
    paramss = 0.7
    L = newdictdf['y'] >= (newdictdf['y'].max()*paramss)
    newdictdf['TF'] = L                                       
    ax[num,1].plot(xm,ym,'o', x_Me, y_Me)
    ax[num,1].set_title("lined up by how long it maintains {0} , Title = {1}".format(num,iname))
for num, iname in enumerate(list(ordinary_method_list.head(10).reset_index()['Track Name'])):
    testRickey = spo[spo['Track Name'] ==iname].groupby('Date').sum()
    testRickey = testRickey.reset_index()
    susul2 = testRickey.loc[:,['Date','Streams']]
    susul2['Ra'] = range(len(susul2))
    ym = np.array(susul2['Streams'])
    xm = np.array(susul2['Ra'])
    zm = np.polyfit(xm, ym, 30)                                                      
    fm = np.poly1d(zm)                      
    x_Me = np.linspace(xm[0], xm[-1], len(testRickey))
    y_Me = fm(x_Me)                         
    newdict = { 'x' : x_Me, 'y' : y_Me}                 
    newdictdf = pd.DataFrame(newdict)  
    paramss = 0.7
    L = newdictdf['y'] >= (newdictdf['y'].max()*paramss)    
    newdictdf['TF'] = L                                        
    ax[num,0].plot(xm,ym,'o', x_Me, y_Me)
    ax[num,0].set_title("lined up by total streaming amount {0} , Title = {1}".format(num,iname))
fig
