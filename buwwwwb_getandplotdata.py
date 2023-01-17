import matplotlib.pyplot as plt

#%matplotlib inline

import numpy as np

import pandas as pd

#import urllib.request

import datetime as dt

pd.options.mode.chained_assignment = None



def plotdf(df, startdate,enddate):

     

    plt.rcParams['figure.figsize'] = [20,15] 

    plt.rcParams['figure.dpi'] = 150

    colortable=['b', 'cornflowerblue','g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

    dfloc=df.loc[startdate:enddate]

    dflen=len(df.columns)

    fig, ax=plt.subplots(dflen)

    #ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    for i, col in enumerate(dfloc.columns):

        plt.subplot(dflen,1,i+1)

        #print(col.find('_p'))

        if col.find('_p')==-1:

            dfloc[col].plot(sharex=plt.gca(),color=colortable[i],label=col,linewidth=1,grid=True)

        else:

            dfloc[col].plot(sharex=plt.gca(),color=colortable[i],marker='|',label=col,grid=True,linestyle=None)

        

        plt.legend(loc="lower left")

    #plt.savefig('temp.png', dpi=fig.dpi)

        



def getdf_frommulticsv(ndays,link="https://uni-wuppertal.sciebo.de/s/rEFPb7PQd3yTMNV/download?path=%2F&files=",fileend='.txt'):

    

    fname=dt.datetime.today().strftime('%Y%m%d')+fileend

    filelink=link+fname

    

    df=pd.read_csv(filelink,index_col=None, sep=';',header=None)

    #todaystr=datetime.today().strftime('%Y%m%d')

    i=1

    while i<ndays:

        datestr=(dt.datetime.today()-dt.timedelta(days=i)).strftime('%Y%m%d')

        #print(datestr)

        fname=datestr+fileend

        filelink=link+fname

        try:

            dfi=pd.read_csv(filelink,index_col=None, sep=';',header=None)

            df=df.append(dfi)

        except:

            print(fname+' not found on File Server')

        i+=1

    df[0] = pd.to_datetime(df[0],format="%d.%m.%Y %H:%M:%S")

    del df[8]

    df[3][df[2]<100]=np.NaN #conditional set values to NaN

    df[2][df[2]<100]=np.NaN

    df[1][df[2]<100]=np.NaN

    df[1][df[1]<0]=np.NaN

    df.columns = ['date','r1min','mabs','rsum','T','H','p','U']

    df.set_index('date', inplace=True)

    df=df.loc[df.index.notnull()]

    df=df.sort_index()

    return(df)
#AUSWAHL

ndays=7 #Anzahl der Tage die im Datenframe abgebildetet werden sollen

#__________________________________________________________________



df=getdf_frommulticsv(ndays)

df.head()
df=df.resample("1min").asfreq() #erstellt für jede Minute eine Messwertzeile und übernimmt vorhandene Werte, Lücken erhalten den Wert: NAN

df['nan']=np.where((df.isnull().all(1)),1,0)  #legt eine Spalte 'nan' an: 1 kein Messwert in Minutenreihe, 0 Messwerte vorhanden

df['nanr']=np.where((df['mabs'].notna()==0)&(df['nan']==0),1,0)  #Spalte nanr_p, regenwert fehlt: 1; regenwert vorhanden oder ganze Zeile fehlt:0

#df['nanr_sum']=df.nanr.groupby(df.index.date).cumsum()

cols=['nan','nanr']

df['nan_sum']=df[cols].sum(axis=1).groupby(df.index.date).cumsum()  #Summenlinie beider nan-Werte mit täglichem Reset

#df.tail()

df['nan_sum'].groupby(df.index.date).max()
#darstellung der ganzen Zeitspanne

startdate=str(df.index[0])

enddate=str(df.index[-1])

plotdf(df,startdate,enddate)

plt.savefig('test.png')

#Darstellung mit definiertem Anfang/Ende

startdate="2020-02-26 00:00:00"

enddate="2020-02-26 18:00:00"

plotdf(df,startdate,enddate)
df.to_csv('mycsv_file.csv',';')
