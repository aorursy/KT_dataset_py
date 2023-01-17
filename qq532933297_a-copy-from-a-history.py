# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# ------------------------------------
import zipfile
from time import time
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

#latitude and longitude of map data
lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]
    
drive = '../input/'
z = zipfile.ZipFile(drive+'train.csv.zip')
train = pd.read_csv(z.open('train.csv'))
print(train)

#get a unique list of categories
cats = list(set(train.Category))
mapdata = np.loadtxt(drive+"sf_map_copyright_openstreetmap_contributors.txt")

#turn strings into dates
dates = []
datesAll = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in train.Dates])

 #set up pandas
startDate = (np.min(datesAll)).date()
endDate = (np.max(datesAll)).date()
alldates = pd.bdate_range(startDate, endDate, freq="m")
dayDF = pd.DataFrame(np.NAN, index=alldates, columns=['x'])
subCats = ['KIDNAPPING','PROSTITUTION','VEHICLE THEFT','LOITERING','SUICIDE','FORGERY/COUNTERFEITING','DRUNKENNESS','DRUG/NARCOTIC','LARCENY/THEFT']

#pdf_pages = PdfPages('crimeData.pdf')
pLoop = 1
for cat in cats:
    saveFile = cat+'.png'
    print(saveFile)
    #just subset for display purposes
    if cat in subCats:
        try:
            fig = plt.figure(figsize = (11.69, 8.27))
            plt.title(cat)
            #plot image
            ax = plt.subplot(2,2,1)
            ax.imshow(mapdata, cmap=plt.get_cmap('gray'),extent=lon_lat_box)
            lineNum = 0
            crime  = cat
            Xcoord = (train[train.Category==cat].X).values
            Ycoord = (train[train.Category==cat].Y).values
            dates = datesAll[np.where(train.Category==cat)]
            Z = np.ones([len(Xcoord),1])
            
            #create dataframe
            df = pd.DataFrame([ [ Z[row][0],Xcoord[row],Ycoord[row]  ] for row in range(len(Z))],
                              index=[dates[row] for row in range(len(dates))],
                              columns=['z','xcoord','ycoord']) 
            
            #resample to sum by month
            df2 = df.resample('m', how='sum')    
            #create uniform time series
            allTimes = dayDF.join(df2).drop('x', axis=1).fillna(0)
            
            movAv = pd.rolling_mean(allTimes['z'],window=12,min_periods=1)
            
            #kde plot by year
            kdeMaxX = []
            kdeMaxY = []
            
            for yLoop in range(2003,2004):
                allData2 = df[(df.index.year == yLoop)]  
                kde = scipy.stats.gaussian_kde(np.array(allData2['xcoord']))
                density = kde(np.array(allData2['xcoord']))
                kde2 = scipy.stats.gaussian_kde(np.array(allData2['ycoord']))
                density2 = kde2(np.array(allData2['ycoord']))
                kdeMaxX.append((allData2['xcoord'][density==np.max(density)]).values[0])
                kdeMaxY.append((allData2['ycoord'][density2==np.max(density2)]).values[0])
            
            #create a quiver plot to show movement of centre of KDE per year
            kdeOut = sns.kdeplot(np.array(allData2['xcoord']), np.array(allData2['ycoord']),shade=True, cut=10, clip=clipsize,alpha=0.5)
            kdeMaxX = np.array(kdeMaxX)
            kdeMaxY = np.array(kdeMaxY)
            plt.quiver(kdeMaxX[:-1], kdeMaxY[:-1], kdeMaxX[1:]-kdeMaxX[:-1], kdeMaxY[1:]-kdeMaxY[:-1], scale_units='xy', angles='xy', scale=1,color='r')
 
            #heatmap to look how data varies by day of week
            ax = plt.subplot(2,2,2)
            heatData = []
            yLoopCount=0
            yearName = []
            weekName = ['mon','tue','wed','thu','fri','sat','sun']
            yearName = ['2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']
            for yLoop in range(2003,2015):
                heatData.append([])
                for dLoop in range(7):
                    allData2 = df[(df.index.year == yLoop) & (df.index.weekday == dLoop)]
                    heatData[yLoopCount].append(sum(allData2['z'].values))
                yLoopCount+=1 
               
            #normlise
            heatData = np.array(heatData)/np.max(np.array(heatData))
            sns.heatmap(heatData, annot=True,xticklabels=weekName,yticklabels=yearName);
                
            plt.title(cat)
            plt.savefig(saveFile)
                #pdf_pages.savefig(fig)
        except:
            print("error: " + cat)