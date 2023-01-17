# This Python 3 environment comes with many helpful analytics libraries installed



import os



import urllib, base64

import datetime as dt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



print( os.listdir("../input"))
# Select Input File

pathFile = "../input/" + 'travels_Kln_19.06.16.log'



df = pd.read_csv( pathFile, engine='python', names=['procress', 'content'], quoting=3, skip_blank_lines=True, sep="DEBUG:" )

df.head()
df =  df[df['content'].str.contains("None") == False ]

dfR = df[df['content'].str.contains("RESULT_ROW ")]

dfRw = dfR['content'].str.replace('RESULT_ROW ', '')

dfA = df['procress']



# Split content columns values and rename columns

dfT = dfRw.str.split('|', expand=True)

dfT.columns = ['TAA','TA','TIN','TIR','TSI','TIM','TIL','TIRE','TIP','TIT','TID','TSC']

for col in dfT.columns:

    dfT[col] = dfT[col].str.replace( col + '-', '')
dfT.head()
# Report infos: city, city-code, date, week-day

cc = dfT.iloc[3]['TSC'].split('%23')

cityCode = cc[1]

cityName = urllib.parse.unquote(cc[0])



cd = dfA.iloc[1].split(' ')

reportDate = cd[0]

reportDayWeekNum =  dt.datetime.strptime( cd[0], '%Y-%m-%d').weekday() # 0 Monday - 6 Sunday

reportDayWeekName =  dt.datetime.strptime( cd[0], '%Y-%m-%d').strftime("%A")
# Clean data: train name, delay to int



def calculateDelay( row ):

    delay = 0

    if row['TA'] is not '':

        taHour = dt.datetime.strptime( str(row['TA']), '%H:%M')

        titHour = dt.datetime.strptime( str(row['TIT']), '%H:%M')

        delay = divmod( ( taHour - titHour ).total_seconds() , 60)[0]  # delay in minutes

    return int(delay)



dfT['TAc'] = dfT.apply( calculateDelay, axis=1 )

dfT =  dfT.join(dfA)

dfT['TIN'] = dfT['TIN'].str.replace('  ', '')
dfSort = dfT.sort_values(['TIT', 'TIN'], ascending=[1, 1])

dfSort.head(10)
# Get list of unique trains refs

dfF = pd.DataFrame( {

    'train': dfSort['TIN'], 

    'hour': dfSort['TIT'], 

    'direction': dfSort['TIM'],

    'delay': dfSort['TAc']

}) 

dfFN = dfF.drop_duplicates(subset=['train','hour','direction'])

#dfFN = dfFN.dropna(subset=['R1'])



# Remove S-bahn

dfFO  = dfFN[ ~(dfFN["train"].str.contains("S ")) ]

dfFO = dfFO.sort_values(['delay'], ascending=[0])

dfFO = dfFO.reset_index()

dfFO.head(25)
grp = dfFO.groupby( by=[ dfFO.hour.map( lambda x : (    int( str( int(x.replace(':','')) )[:3] + '0')     )) , dfFO.delay  ] )

dfG =  grp.size().to_frame('trains')

dfGA = dfG.reset_index()

dfGA.head(25)

dfGA.describe()
dfGA.plot( x='hour', y='delay', style=['o','rx'] )
plt.figure(figsize=(20,5))



plt.bar( dfGA.hour, dfGA.trains, width=8 )

plt.xlabel('Hours', fontsize=12)

plt.ylabel('Trains', fontsize=12)

plt.yticks(np.arange(0, 20, step=1), fontsize=12, rotation=0 )

plt.xticks(np.arange(1400, 2000, step=10), fontsize=12, rotation=90 )

plt.title('Delayed trains - Köln 19.06.16')

plt.show()
sumTrains = len(dfFO)

#COM: prepare report result object

reportCityDateTrains = {

    'cityCode'  : cityCode,

    'cityName'  : cityName,

    'reportDate': reportDate,

    'reportDayWeek': reportDayWeekName,

    'sumTrains' : sumTrains

}



print(reportCityDateTrains)