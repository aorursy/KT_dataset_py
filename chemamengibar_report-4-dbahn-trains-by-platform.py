# Imports

import sys

import os

import urllib, base64

import datetime as dt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.colors



from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define input files

datasetFolder = '../input//'



files = [

    '19.07.19_travels_Bonn.csv',

    '19.07.19_travels_Koblenz.csv',

    '19.07.19_travels_Koeln.csv',

]

def loadFile( _datasetFolder, _fileName ):

    fullPathFile = _datasetFolder + _fileName

    df = pd.read_csv( fullPathFile, engine='python', quoting=1, skip_blank_lines=True,

                     sep=",", quotechar='"',

                     encoding ='utf-8', dtype={'TSI': 'object'}

                    )

    return df.replace(np.nan, '', regex=True)
dfsDay = [

    loadFile( datasetFolder, files[0]),

    loadFile( datasetFolder, files[1]),

    loadFile( datasetFolder, files[2])

]
dfsDay[0].head(1)
# Select the index for the next steps

cityIndex = 1
# Extract City name and code

fullCityCode = dfsDay[ cityIndex ].iloc[0]['TSC']

cc = fullCityCode.split('%23')

cityCode = cc[1]

cityName = urllib.parse.unquote(cc[0])



cityCode, cityName
# Extract report date

reportDate = '20' + dfsDay[ cityIndex ].iloc[0]['TID']

reportDayWeekNum =  dt.datetime.strptime( reportDate, '%Y.%m.%d').weekday() # 0 Monday - 6 Sunday

reportDayWeekName =  dt.datetime.strptime( reportDate, '%Y.%m.%d').strftime("%A")



reportDate, reportDayWeekNum, reportDayWeekName
def extractDataColumns( _df ):

    dfF0 = _df.sort_values(['TIT', 'TIN'], ascending=[1, 1])

    dfF1 = pd.concat( [

        dfF0['TIN'] ,

        dfF0['TIP'],

        dfF0['TIT'].apply( lambda t: 

            int( dt.datetime.strptime( t, '%H:%M' ).strftime('%H') )

        ),

        dfF0['TIT']

    ],axis=1, keys=['train', 'platform', 'hour','time'] )

    return dfF1[ ~(dfF1["train"].str.contains("S")) ].reset_index(drop=True)
dfBaseSorted = [

    extractDataColumns(dfsDay[0]),

    extractDataColumns(dfsDay[1]),

    extractDataColumns(dfsDay[2])

]



dfBaseSorted[0].head(2)
def customGroup1(x):

    if x.name == 'train':

        return x.iloc[0]

    elif  x.name == 'platform':

        return x.iloc[0]

    else:

        return x.iloc[0]





f = { "data" : lambda x: customGroup1(x) }



dfG = dfBaseSorted[cityIndex]

dfGA = dfG.groupby(['train','hour'], as_index=True)[['platform','time']].agg(f)

dfGA.head(5)
dfGAM = pd.DataFrame({

    'train': list(zip(*dfGA.index))[0],

    'hour': list(zip(*dfGA.index))[1],

    'platform': list(zip(*dfGA.values))[0],

    'time': list(zip(*dfGA.values))[1]



})

dfGAM = dfGAM.sort_values(['time','platform'], ascending=[1,1]).reset_index(drop=True)

dfGAM.head()
g = dfGAM.groupby(['platform','hour'], as_index=True)

g.describe().head(5)
def f( x ):

    if x.name == 'train':

        l = x.items()

        #return list(zip(*x.items()))[1]

        return ','.join([list(t) for t in zip(*l)][1])

    if x.name == 'time':

        return x.count()

    

'''

https://stackoverflow.com/questions/12974474/how-to-unzip-a-list-of-tuples-into-individual-lists/12974504

'''

dfGAMSG = dfGAM.groupby(['platform','hour'], as_index=True).agg( f ) #{'time' : ['sum', 'min', 'max']}

dfGAMSG.rename(columns={'time':'count'}, inplace=True)

dfGAMSG.head(15)
platforms = list(dfGAMSG.index.unique().levels[0])

hours = list(dfGAMSG.index.unique().levels[1])

print(platforms,'\n',hours)
dGrid = [] # data list

mGrid = [] # metadata list

for ip, p in enumerate(platforms):

    dRow = []

    mRow=[]

    for ih, h in enumerate(hours):

        dfQ = dfGAMSG.query('hour == @h & platform == @p')['count']

        v = dfQ.values[0] if not dfQ.empty else 0

        dRow.append(v)

        mRow.append(['p'+p,'h'+str(h)])

    dGrid.append(dRow)

    mGrid.append(mRow)

#print( dGrid, mGrid ) 
flatDGrid = [item for sublist in dGrid for item in sublist]

maxTrains = max(set(flatDGrid))

minTrains = 0
plt.style.use('seaborn')



fig, axs = plt.subplots(1, 4, figsize=(17,3)) #subplots(2,2,..) -> axs[0,0]

fig.suptitle('Trains by platform-hour in ' + cityName + ' - ' + reportDate, y=0.98, fontsize=14)

fig.subplots_adjust( top=0.75, wspace=0.5)



#0

im0 = axs[0].imshow(dGrid, cmap='hot', interpolation='nearest', aspect="auto")

axs[0].set_xticks(np.arange(0, len(hours), 1));

axs[0].set_yticks(np.arange(0, len(platforms), 1));

axs[0].set_xticklabels(hours);

axs[0].set_yticklabels(platforms);

axs[0].set_title('no threshold ')

axs[0].grid(False) 



divider0 = make_axes_locatable(axs[0])

cax0 = divider0.append_axes('right', size='5%', pad=0.05)

fig.colorbar(im0, cax=cax0, orientation='vertical')





#1

max_threshold = 5

min_threshold = 2

im1 = axs[1].imshow(dGrid, cmap='Accent', vmin=min_threshold, vmax=max_threshold, aspect="auto")

axs[1].set_xticks(np.arange(0, len(hours), 1));

axs[1].set_yticks(np.arange(0, len(platforms), 1));

axs[1].set_xticklabels(hours);

axs[1].set_yticklabels(platforms);

axs[1].set_title('threshold ' + str(min_threshold) + '-' + str(max_threshold))

axs[1].grid(False) 



divider1 = make_axes_locatable(axs[1])

cax1 = divider1.append_axes('right', size='5%', pad=0.05)

fig.colorbar(im1, cax=cax1, orientation='vertical')





#2

listColors = [[150, 255, 224],[150, 255, 224],[150, 255, 224],[150, 255, 224],

              [150, 255, 224],[42, 219, 166],[235, 64, 52]]

rgb=np.array(listColors)/255.

ccmapRgb = matplotlib.colors.ListedColormap( rgb )



max_threshold = 5

min_threshold = 2

im2 = axs[2].imshow(dGrid, cmap=ccmapRgb, aspect="auto")

axs[2].set_xticks(np.arange(0, len(hours), 1));

axs[2].set_yticks(np.arange(0, len(platforms), 1));

axs[2].set_xticklabels(hours);

axs[2].set_yticklabels(platforms);

axs[2].set_title('custom cmap')

axs[2].grid(False) 



divider2 = make_axes_locatable(axs[2])

cax2 = divider2.append_axes('right', size='5%', pad=0.05)

fig.colorbar(im2, cax=cax2, orientation='vertical')





#3

ccmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['red','blue'])



max_threshold = 5

min_threshold = 2

im3 = axs[3].imshow(dGrid, cmap=ccmap, vmin=min_threshold, vmax=max_threshold, aspect="auto")

axs[3].set_xticks(np.arange(0, len(hours), 1));

axs[3].set_yticks(np.arange(0, len(platforms), 1));

axs[3].set_xticklabels(hours);

axs[3].set_yticklabels(platforms);

axs[3].set_title('threshold ' + str(min_threshold) + '-' + str(max_threshold))

axs[3].grid(False) 



divider3 = make_axes_locatable(axs[3])

cax3 = divider3.append_axes('right', size='5%', pad=0.05)

fig.colorbar(im3, cax=cax3, orientation='vertical')



plt.show()