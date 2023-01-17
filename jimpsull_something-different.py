#The standard imports Kaggle give you when you start a kernel
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import warnings
from matplotlib import pyplot as plt
import copy
import scipy.stats as ss
#from astropy.time import Time
#import fbprophet as fbp
#import FATS
#import cesium

print('Getting Base (meta) dataFrame')
#bdf=pd.read_csv('../input/test_set_metadata.csv')
bdf=pd.read_csv('../input/training_set_metadata.csv')

print(bdf.shape)
#print('Postponing getting raw (lightcurve) dataFrame')
rdf=pd.read_csv('../input/training_set.csv')
#print(rdf.shape)
print(rdf.shape)
lep=2
hep=5
# originally planned on 2, 4 based on DDF data
# based on all data going with 2, 5

def filterRawDf(ordf, lowEngPb=lep, highEngPb=hep):
    #frdf=rdf[rdf['passband'] in [2,5]]
    rdf=copy.deepcopy(ordf)
    filterLow = rdf.loc[:,'passband']==lowEngPb
    filterHigh = rdf.loc[:,'passband']==highEngPb
    filterPb = filterLow | filterHigh
    frdf=rdf.loc[filterPb,:]
    return frdf

frdf = filterRawDf(rdf)
frdf.shape

#t=Time(frdf.loc[:,'mjd'])
#print(t.isot)
#frdf=frdf.rename(columns={'mjd':'ds', 'flux':'y'})
#get light curve data for one object_id from the raw data
#this can be used with either rdf or filtered raw data (frdf)
def getLCDF(ordf, objid, show=False):
    rdf=copy.deepcopy(ordf)
    lcdf=rdf[rdf['object_id']==objid]
    if show:
        plt.plot(lcdf.loc[:,'mjd'],lcdf.loc[:,'flux'])
        plt.show()

        
    return lcdf
        
elcdf=getLCDF(frdf,615, show=True)
print(elcdf.shape)
#print(frdf.shape)
elcdf=getLCDF(frdf, 1019335, show=True)
print(elcdf.shape)
def divideLcdf(elcdf, ddf, lep=2, hep=5):
    lcdf=copy.deepcopy(elcdf)
    #this simple date cutting works on the ddf objects
    if ddf:
        minDate=np.min(elcdf.loc[:,'mjd'])
        maxDate=np.max(elcdf.loc[:,'mjd'])

        halfPoint=np.average([minDate, maxDate])
        firstCut=np.average([minDate, halfPoint])
        secondCut=np.average([halfPoint, maxDate])
        minDate=np.min(elcdf.loc[:,'mjd'])
        maxDate=np.max(elcdf.loc[:,'mjd'])

        halfPoint=np.average([minDate, maxDate])
        firstCut=np.average([minDate, halfPoint])
        secondCut=np.average([halfPoint, maxDate])

        #early
        efilter=elcdf.loc[:,'mjd']<=firstCut
        #late
        lfilter=elcdf.loc[:,'mjd']>=secondCut
        #mid
        mfilter=(efilter | lfilter)==False
    
        edf=elcdf.loc[efilter]
        mdf=elcdf.loc[mfilter]
        ldf=elcdf.loc[lfilter]
        
        ledf = edf[edf['passband']==lep]
        hedf = edf[edf['passband']==hep]
        lmdf = mdf[mdf['passband']==lep]
        hmdf = mdf[mdf['passband']==hep]
        lldf = ldf[ldf['passband']==lep]
        hldf = ldf[ldf['passband']==hep]
    
    #using the datecutting method often leads to zero population sizes with non-ddf objects
    else:
        
        lowdf=elcdf[elcdf['passband']==lep]
        highdf=elcdf[elcdf['passband']==hep]
        lenLow=lowdf.shape[0]
        lenHigh=highdf.shape[0]
        
        minSizeLow = int(lenLow / 3)
        minSizeHigh = int(lenHigh / 3)
        
        lldf=lowdf.nlargest(minSizeLow, 'mjd')
        hldf=highdf.nlargest(minSizeHigh, 'mjd')
        ledf=lowdf.nsmallest(minSizeLow, 'mjd')
        hedf=highdf.nsmallest(minSizeHigh, 'mjd')
        lmdf=lowdf.nlargest(lenLow-minSizeLow, 'mjd').nsmallest(lenLow-2*minSizeLow, 'mjd')
        hmdf=highdf.nlargest(lenHigh-minSizeHigh, 'mjd').nsmallest(lenHigh-2*minSizeHigh, 'mjd')
    
    return ledf, hedf, lmdf, hmdf, lldf, hldf

ledf, hedf, lmdf, hmdf, lldf, hldf=divideLcdf(elcdf, 0)    
print(ledf.shape)
print(lmdf.shape)
print(lldf.shape)
print(hedf.shape)
print(hmdf.shape)
print(hldf.shape)
def getSubPopFeats(pbdf, outSig=3.0):
    
    average=np.average(pbdf.loc[:,'flux'])
    median=np.median(pbdf.loc[:,'flux'])
    stdev=np.std(pbdf.loc[:,'flux'])
    maxflux=np.max(pbdf.loc[:,'flux'])
    minflux=np.min(pbdf.loc[:,'flux'])
    stdflerr=np.std(pbdf.loc[:,'flux_err'])
    medflerr=np.median(pbdf.loc[:,'flux_err'])
    
    #We want a means to extract the rate of decay or rise from minima or maxima
    #This is grabbing within the population
    #We also will look between populations
    maxmjd=np.max(pbdf[pbdf['flux']==maxflux].loc[:,'mjd'])
    minmjd=np.max(pbdf[pbdf['flux']==minflux].loc[:,'mjd'])
    
    #at what date does the max occur?
    aftmaxdf=pbdf[pbdf['mjd']>maxmjd]
    
    #if there are data points after the max, what is the value and date of the lowest?
    if aftmaxdf.shape[0]>0:
        minaft=np.min(aftmaxdf.loc[:,'flux'])
        aftminmjd=np.min(aftmaxdf[aftmaxdf['flux']==minaft].loc[:,'mjd'])
        #(val at t0 - val at t1) / (t0 - t1) sb neg
        decaySlope=(maxflux-minaft)/(maxmjd-aftminmjd)
    
    else:
        decaySlope=0
        
    aftmindf=pbdf[pbdf['mjd']<minmjd]
    if aftmindf.shape[0]>0:
        maxaft=np.max(aftmindf.loc[:,'flux'])
        aftmaxmjd=np.max(aftmindf[aftmindf['flux']==maxaft].loc[:,'mjd'])
        #(val at t0 - val at t1) / (t0 - t1) sb pos
        riseSlope=(minflux - maxaft)/(aftmaxmjd-minmjd)
    
    else:
        riseSlope=0
        
    return average, stdev, median, medflerr, stdflerr, maxflux, \
            maxmjd, decaySlope, minflux, minmjd, riseSlope

a,b,c,d,e,f,g, h,i,j,k=getSubPopFeats(hmdf)
print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)
print(h)
print(i)
print(j)
print(k)

def processLc(objid, elcdf, ddf, lep=2, hep=5):
    
    lcdf=copy.deepcopy(elcdf)
    
    #feature borrowed from Grzegorz Sionkowski (../sionek)
    #dt[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]
    #detectMjds=elcdf[elcdf['detected']==1].loc[:,'mjd']
    #deltaDetect=np.max(detectMjds) - np.min(detectMjds)
    
    #divide the incoming light curve to 6 subpopulations
    ledf, hedf, lmdf, hmdf, lldf, hldf=divideLcdf(lcdf, ddf,lep=lep, hep=hep)
    #return average, stdev, median, medflerr, stdflerr, maxflux, \
    #        maxmjd, decayslope, minflux, minmjd, riseSlope
    
    leavg, lestd, lemed, lemfl, lesfl, lemax, lemxd, ledsl, lemin, lemnd, lersl=getSubPopFeats(ledf)
    heavg, hestd, hemed, hemfl, hesfl, hemax, hemxd, hedsl, hemin, hemnd, hersl=getSubPopFeats(hedf)
    lmavg, lmstd, lmmed, lmmfl, lmsfl, lmmax, lmmxd, lmdsl, lmmin, lmmnd, lmrsl=getSubPopFeats(lmdf)
    hmavg, hmstd, hmmed, hmmfl, hmsfl, hmmax, hmmxd, hmdsl, hmmin, hmmnd, hmrsl=getSubPopFeats(hmdf)
    llavg, llstd, llmed, llmfl, llsfl, llmax, llmxd, lldsl, llmin, llmnd, llrsl=getSubPopFeats(lldf)
    hlavg, hlstd, hlmed, hlmfl, hlsfl, hlmax, hlmxd, hldsl, hlmin, hlmnd, hlrsl=getSubPopFeats(hldf)
    
    
    feats= [objid, leavg, lestd, lemed, lemfl, lesfl, lemax, 
            lemxd, ledsl, lemin, lemnd, lersl,
            heavg, hestd, hemed, hemfl, hesfl, hemax, hemxd,
            hedsl, hemin, hemnd, hersl,
            lmavg, lmstd, lmmed, lmmfl, lmsfl, lmmax, lmmxd,
            lmdsl, lmmin, lmmnd, lmrsl,
            hmavg, hmstd, hmmed, hmmfl, hmsfl, hmmax, hmmxd, 
            hmdsl, hmmin, hmmnd, hmrsl,
            llavg, llstd, llmed, llmfl, llsfl, llmax, llmxd, 
            lldsl, llmin, llmnd, llrsl,
            hlavg, hlstd, hlmed, hlmfl, hlsfl, hlmax, hlmxd, 
            hldsl, hlmin, hlmnd, hlrsl]
    
    return feats

feats=processLc(1019335, elcdf, 0)

print(feats)
    
from io import StringIO
from csv import writer 
import time

def writeAChunk(firstRecord, lastRecord, bdf, frdf, statusFreq=500):
    output = StringIO()
    csv_writer = writer(output)

    fdf=pd.DataFrame(columns=['objid', 'leavg', 'lestd', 'lemed', 'lemfl', 'lesfl', 'lemax', 
                'lemxd', 'ledsl', 'lemin', 'lemnd', 'lersl',
                'heavg', 'hestd', 'hemed', 'hemfl', 'hesfl', 'hemax', 'hemxd',
                'hedsl', 'hemin', 'hemnd', 'hersl',
                'lmavg', 'lmstd', 'lmmed', 'lmmfl', 'lmsfl', 'lmmax', 'lmmxd',
                'lmdsl', 'lmmin', 'lmmnd', 'lmrsl',
                'hmavg', 'hmstd', 'hmmed', 'hmmfl', 'hmsfl', 'hmmax', 'hmmxd', 
                'hmdsl', 'hmmin', 'hmmnd', 'hmrsl',
                'llavg', 'llstd', 'llmed', 'llmfl', 'llsfl', 'llmax', 'llmxd', 
                'lldsl', 'llmin', 'llmnd', 'llrsl',
                'hlavg', 'hlstd', 'hlmed', 'hlmfl', 'hlsfl', 'hlmax', 'hlmxd', 
                'hldsl', 'hlmin', 'hlmnd', 'hlrsl'])

    theColumns=fdf.columns
    
    csv_writer.writerow(theColumns)
    started=time.time()
    for rindex in range(firstRecord, lastRecord):
        #if you want to monitor progress
        #ddf 18 sec per 100 on my macAir
        #non ddf 25 sec per 100 on my macAir
        if rindex%statusFreq==(statusFreq-1):
            print(rindex)
            print("Processing took {:6.4f} secs, records = {}".format((time.time() - started), statusFreq))
            started=time.time()
            #fdf=pd.merge(fdf, tdf, on='key')
        objid = bdf.loc[rindex,'object_id']
        ddf=bdf.loc[rindex,'ddf']==1
        #ig=bdf.loc[rindex,'hostgal_specz']==0
        lcdf=getLCDF(frdf, objid)
        feats=processLc(objid, lcdf, ddf)
        #fdf.loc[rindex,:]=feats
        csv_writer.writerow(feats)

    output.seek(0) # we need to get back to the start of the BytesIO
    chdf = pd.read_csv(output)
    chdf.columns=theColumns
    
    return chdf

theColumns=['objid', 'leavg', 'lestd', 'lemed', 'lemfl', 'lesfl', 'lemax', 
                'lemxd', 'ledsl', 'lemin', 'lemnd', 'lersl',
                'heavg', 'hestd', 'hemed', 'hemfl', 'hesfl', 'hemax', 'hemxd',
                'hedsl', 'hemin', 'hemnd', 'hersl',
                'lmavg', 'lmstd', 'lmmed', 'lmmfl', 'lmsfl', 'lmmax', 'lmmxd',
                'lmdsl', 'lmmin', 'lmmnd', 'lmrsl',
                'hmavg', 'hmstd', 'hmmed', 'hmmfl', 'hmsfl', 'hmmax', 'hmmxd', 
                'hmdsl', 'hmmin', 'hmmnd', 'hmrsl',
                'llavg', 'llstd', 'llmed', 'llmfl', 'llsfl', 'llmax', 'llmxd', 
                'lldsl', 'llmin', 'llmnd', 'llrsl',
                'hlavg', 'hlstd', 'hlmed', 'hlmfl', 'hlsfl', 'hlmax', 'hlmxd', 
                'hldsl', 'hlmin', 'hlmnd', 'hlrsl']

fdf=pd.DataFrame(columns=theColumns)
chunksize=2616
firstLoop=0
lastLoop=3
loops=lastLoop-firstLoop
veryFirstRow=firstLoop*chunksize
veryLastRow=lastLoop*chunksize-1
for i in range(firstLoop, lastLoop):
    startRow=i*chunksize
    stopRow=(i+1)*chunksize
    chdf=writeAChunk(startRow, stopRow, bdf, frdf, statusFreq=int(chunksize/2))
    fdf= pd.concat([fdf, chdf])
    print(fdf.shape)


fdf=fdf.rename({'objid':'object_id'},axis=1)
bdf.loc[:,'object_id']=bdf.loc[:,'object_id'].astype(str)
fdf.loc[:,'object_id']=fdf.loc[:,'object_id'].astype(str)
#DataFrame.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False)[source]¶
mdf=bdf.merge(fdf, sort=False)
print(mdf.shape)
mdf.head()
def testForOutlier(bdf, energy='high', sigmas=1.0):
    
    if energy=='high':
        valCols=['heavg', 'hmavg', 'hlavg']
        sigCols=['hestd', 'hmstd', 'hlstd']
    else:
        valCols=['leavg', 'lmavg', 'llavg']
        sigCols=['lestd', 'lmstd', 'llstd']
    
    fdf=copy.deepcopy(bdf)
    

    
    fdf.loc[:,energy + 'Energy_transitory_' + str(round(sigmas,1)) + '_TF']=False
    for i in range(len(valCols)):
        fdf.loc[:,'min' + str(valCols[i])] = fdf.loc[:,valCols[i]] - sigmas*fdf.loc[:,sigCols[i]]
        
        fdf.loc[:,'max' + str(valCols[i])] = fdf.loc[:,valCols[i]] + sigmas*fdf.loc[:,sigCols[i]]
    
    for i in range(len(valCols)):
        #fdf.loc[:,'earlySet']=range(fdf.loc[:,'minX100' + str(valCols[0])],fdf.loc[:, 'maxX100' + str(valCols[0])])
        #earlyMaxLessThanMedMin
        for j in range(len(valCols)):
            if j!=i:
                
                maxFailsOverlap=fdf.loc[:,'max' + str(valCols[i])]<fdf.loc[:,'min' + str(valCols[j])]
                minFailsOverlap=fdf.loc[:,'min' + str(valCols[i])]>fdf.loc[:,'max' + str(valCols[j])]
                theValue= (fdf.loc[:,energy + 'Energy_transitory_' + str(round(sigmas,1)) + '_TF'] | minFailsOverlap | maxFailsOverlap)
                #theValue=theValue.astype(str)
                fdf.loc[:,energy + 'Energy_transitory_' + str(round(sigmas,1)) + '_TF']=theValue
                #fdf.loc[:,energy + '_' + str(valCols[i]) + '_' + str(valCols[j])] = str(theValue) + \
                #+ '_' + str(maxFailsOverlap) + '_'+ str(minFailsOverlap)
    for i in range(len(valCols)):
        fdf=fdf.drop('min' + str(valCols[i]), axis=1)
        fdf=fdf.drop('max' + str(valCols[i]), axis=1)
        
    return fdf

energy='high'
sigmas=1.0
fdf=testForOutlier(mdf)
fdf.shape
print(fdf.loc[:,energy + 'Energy_transitory_' + str(round(sigmas,1)) + '_TF'].sum())

sigmas=1.5
fdf=testForOutlier(fdf, energy=energy, sigmas=sigmas)
print(fdf.loc[:,energy + 'Energy_transitory_' + str(round(sigmas,1)) + '_TF'].sum())

energy='low'
sigmas=1.0
fdf=testForOutlier(fdf, energy=energy, sigmas=sigmas)
print(fdf.loc[:,energy + 'Energy_transitory_' + str(round(sigmas,1)) + '_TF'].sum())

sigmas=1.5
fdf=testForOutlier(fdf, energy=energy, sigmas=sigmas)
print(fdf.loc[:,energy + 'Energy_transitory_' + str(round(sigmas,1)) + '_TF'].sum())

print(fdf.shape)
fdf.head()
fdf.loc[:,'outlierString']=fdf.loc[:,'highEnergy_transitory_1.5_TF'].astype(str) + \
                             fdf.loc[:,'highEnergy_transitory_1.0_TF'].astype(str) + \
                             fdf.loc[:,'lowEnergy_transitory_1.5_TF'].astype(str) + \
                             fdf.loc[:,'lowEnergy_transitory_1.0_TF'].astype(str)


def getOutlierScore(row):
    tdict={'TrueTrueTrueTrue':8, 'FalseTrueTrueTrue':7, 'TrueTrueFalseTrue':7,
       'FalseTrueFalseTrue':6, 'FalseFalseTrueTrue':3, 'TrueTrueFalseFalse':3,
       'FalseTrueFalseFalse':3, 'FalseFalseFalseTrue':3, 'FalseFalseFalseFalse':0}
    return tdict[row['outlierString']]

fdf['outlierScore']=fdf.apply(getOutlierScore, axis=1)
    
fdf=fdf.drop('outlierString', axis=1)

#fdf.to_csv('fastestFeatureTableWithTransitoryFlags.csv')
print(fdf.shape)
print(fdf.columns)
print(np.average(fdf.loc[:,'outlierScore']))
print(np.min(fdf.loc[:,'outlierScore']))
print(np.max(fdf.loc[:,'outlierScore']))
print(np.median(fdf.loc[:,'outlierScore']))
fdf['hipd']=0
fdf['hipr']=0
fdf['htpd']=0
fdf['htpr']=0

fdf['lipd']=0
fdf['lipr']=0
fdf['ltpd']=0
fdf['ltpr']=0

outlierFilter=(fdf['outlierScore']>0)
print(outlierFilter.sum())

hipdFilter = (fdf['hmmax']>fdf['hemax']) & (fdf['hmmax']>fdf['hlmax']) & outlierFilter
htpdFilter = (fdf['hemax']>fdf['hmmax']) & (fdf['hemax']>fdf['hlmax']) & outlierFilter
lipdFilter = (fdf['lmmax']>fdf['lemax']) & (fdf['lmmax']>fdf['llmax']) & outlierFilter
ltpdFilter = (fdf['lemax']>fdf['lmmax']) & (fdf['lemax']>fdf['llmax']) & outlierFilter

print(hipdFilter.sum())
print(htpdFilter.sum())
print(lipdFilter.sum())
print(ltpdFilter.sum())


#peak to peak
#these are light curves where the peak was in the middle
fdf.loc[hipdFilter,'hipd']=(fdf.loc[hipdFilter,'hmmax']-fdf.loc[hipdFilter,'hlmax']) / \
     (fdf.loc[hipdFilter,'hmmxd']-fdf.loc[hipdFilter,'hlmxd'])
fdf.loc[lipdFilter,'lipd']=(fdf.loc[lipdFilter,'lmmax']-fdf.loc[lipdFilter,'llmax']) / \
     (fdf.loc[lipdFilter,'lmmxd']-fdf.loc[lipdFilter,'llmxd'])

#these are light curves where the peak was in the beginning
fdf.loc[htpdFilter,'hipd']=(fdf.loc[htpdFilter,'hemax']-fdf.loc[htpdFilter,'hmmax']) / \
     (fdf.loc[htpdFilter,'hemxd']-fdf.loc[htpdFilter,'hmmxd'])
fdf.loc[ltpdFilter,'lipd']=(fdf.loc[ltpdFilter,'lemax']-fdf.loc[ltpdFilter,'lmmax']) / \
     (fdf.loc[ltpdFilter,'lemxd']-fdf.loc[ltpdFilter,'lmmxd'])
fdf.loc[htpdFilter,'htpd']=(fdf.loc[htpdFilter,'hmmax']-fdf.loc[htpdFilter,'hlmax']) / \
     (fdf.loc[htpdFilter,'hmmxd']-fdf.loc[htpdFilter,'hlmxd'])
fdf.loc[ltpdFilter,'ltpd']=(fdf.loc[ltpdFilter,'lmmax']-fdf.loc[ltpdFilter,'llmax']) / \
     (fdf.loc[ltpdFilter,'lmmxd']-fdf.loc[ltpdFilter,'llmxd'])

#print(fdf.loc[lipdFilter,'lipd'])
fdf[outlierFilter].head()
hiprFilter = (fdf['hmmin']<fdf['hemin']) & (fdf['hmmin']<fdf['hlmin']) & outlierFilter
htprFilter = (fdf['hemin']<fdf['hmmin']) & (fdf['hemin']<fdf['hlmin']) & outlierFilter
liprFilter = (fdf['lmmin']<fdf['lemin']) & (fdf['lmmin']<fdf['llmin']) & outlierFilter
ltprFilter = (fdf['lemin']<fdf['lmmin']) & (fdf['lemin']<fdf['llmin']) & outlierFilter

#these are light curves where the peak was in the middle
fdf.loc[hipdFilter,'hipr']=(fdf.loc[hipdFilter,'hmmin']-fdf.loc[hipdFilter,'hlmin']) / \
     (fdf.loc[hipdFilter,'hmmnd']-fdf.loc[hipdFilter,'hlmnd'])
fdf.loc[lipdFilter,'lipr']=(fdf.loc[lipdFilter,'lmmin']-fdf.loc[lipdFilter,'llmin']) / \
     (fdf.loc[lipdFilter,'lmmnd']-fdf.loc[lipdFilter,'llmnd'])

#these are light curves where the peak was in the beginning
fdf.loc[htpdFilter,'hipr']=(fdf.loc[htpdFilter,'hemin']-fdf.loc[htpdFilter,'hmmin']) / \
     (fdf.loc[htpdFilter,'hemnd']-fdf.loc[htpdFilter,'hmmnd'])
fdf.loc[ltpdFilter,'lipr']=(fdf.loc[ltpdFilter,'lemin']-fdf.loc[ltpdFilter,'lmmin']) / \
     (fdf.loc[ltpdFilter,'lemnd']-fdf.loc[ltpdFilter,'lmmnd'])
fdf.loc[htpdFilter,'htpr']=(fdf.loc[htpdFilter,'hmmin']-fdf.loc[htpdFilter,'hlmin']) / \
     (fdf.loc[htpdFilter,'hmmnd']-fdf.loc[htpdFilter,'hlmnd'])
fdf.loc[ltpdFilter,'ltpr']=(fdf.loc[ltpdFilter,'lmmin']-fdf.loc[ltpdFilter,'llmin']) / \
     (fdf.loc[ltpdFilter,'lmmnd']-fdf.loc[ltpdFilter,'llmnd'])

fdf[outlierFilter].head()
fdf.to_csv('newTrainFeatureOutputUnprocessed.csv')
print(fdf.shape)