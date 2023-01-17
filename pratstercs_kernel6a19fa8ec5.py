import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Input data files are available in the "../input/" directory.
races = pd.read_csv( '../input/races.csv')
raceData = races[['raceId','year','name']]
results = pd.read_csv( '../input/results.csv')
allData = raceData.set_index('raceId').join(results.set_index('raceId'))
allData = allData[['year','name','position','time','milliseconds']]
allData = allData.query( 'position in [1,2]')
row = allData.iloc[1]
row
datetime.timedelta(milliseconds=row['milliseconds'])
def getDelta( row ):

    try:

        return datetime.timedelta(milliseconds=row['milliseconds']) - datetime.timedelta(milliseconds=allData.query( f'year == "{row["year"]}" & name == "{row["name"]}" & position == 1' ).iloc[0]['milliseconds'])

    except:

        return None
allData['margin'] = allData.apply( getDelta, axis=1)
allData = allData.query( 'position == 2')
margins = allData[['margin','year','name']]
margins.dropna()
byYear = margins.groupby('year')
def agger( series ):

    s2 = series.dropna()

    return (s2.sum() / len(s2)).total_seconds()

    

aggregrate = byYear.agg( agger )
aggregrate
plot = aggregrate.plot( figsize=(10,10), grid=True )

plot.set_xlim(1950,2019)

plot.set_ylim(0,120)

plot.grid(True,'major')

plot.grid(True,'minor')