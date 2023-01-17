import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# general variables used
interval = 7

# data are „Fallzahlen in Deutschland“ of the Robert Koch Institute (RKI), Germany:
# https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Fallzahlen.html
# license-ID: „dl-de-by-2.0"
# license for 'Open Data Datenlizenz Deutschland – Namensnennung – Version 2.0':
#             https://www.govdata.de/dl-de/by-2-0

# location:
# remote (newest data): https://opendata.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0.csv
# local: ../input//dd4580c810204019a7b8eb3e0b329dd6_0.csv
corona = pd.read_csv('https://opendata.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0.csv',
                    index_col='ObjectId', parse_dates=['Meldedatum', 'Refdatum'],
                    usecols=lambda column : column not in ["IdBundesland", "Altersgruppe2", "Geschlecht", "NeuerFall", "NeuerTodesfall", "NeuGenesen"])

corona.dropna(axis=1, inplace=True)
corona.head(20)
# We don't need the column 'Datenstand'. It seems to have only informational value
corona.drop(['Datenstand'], axis=1, inplace=True)
corona.info()
# Define lists for all age classes (0-4, 5-14, 15-34, 35-59, 60-79, 80+)
A00 = []
A05 = []
A15 = []
A35 = []
A60 = []
A80 = []

def addWeightToAltersgruppe(weight, altersgruppe):
    unknown = 0
    
    if weight < 0: print ("negative weight found: ", weight)
    
    if altersgruppe == 'A00-A04':
        A00.append(weight)
    elif altersgruppe == 'A05-A14':
        A05.append(weight)
    elif altersgruppe == 'A15-A34':
        A15.append(weight)
    elif altersgruppe == 'A35-A59':
        A35.append(weight)
    elif altersgruppe == 'A60-A79':
        A60.append(weight)
    elif altersgruppe == 'A80+':
        A80.append(weight)
    else:
        unknown += 1
        
# Mark entries where Meldedatum-Erkrankungsdatum > 30 as unknown
def correctErkrankungsDatum(IstErkr, MeldeDatum, Refdatum, Altersgruppe):
    if (IstErkr == 0):
        return 0
    else:
        timeDelta = MeldeDatum - Refdatum
        if timeDelta.days > 30 or timeDelta.days < 0:
            # Erkrangusdatum of this row is too far away from MeldeDatum or negative
            return 0
        else:
            # collect delta of MeldeDatum and ReferenzDatum
            addWeightToAltersgruppe(timeDelta.days, Altersgruppe)
            return 1

coronaErkrBegin = corona[corona['IstErkrankungsbeginn'] == 1]
print("Amount of data with known beginning date of illness: ", (coronaErkrBegin.shape[0] * 100)/corona.shape[0] ,"%")

corona['IstErkrankungsbeginn'] = corona.apply(lambda x: correctErkrankungsDatum(
                x['IstErkrankungsbeginn'], x['Meldedatum'], x['Refdatum'], x['Altersgruppe']), axis=1)
# calculate the Meanvalue for each of the collected Axx values
M00 = round(pd.Series(A00).mean())
M05 = round(pd.Series(A05).mean())
M15 = round(pd.Series(A15).mean())
M35 = round(pd.Series(A35).mean())
M60 = round(pd.Series(A60).mean())
M80 = round(pd.Series(A80).mean())

# empty the long Axx lists because we don't need them any longer
A00 = []
A05 = []
A15 = []
A35 = []
A60 = []
A80 = []

print("Delta of all six age classes: ", M00, M05, M15, M35, M60, M80)
# Adjust entries with unknown sickness date with M00 .. M80
# ==> imputedDate (imputiertes Datum)

def addMeanErkrankungsDatum(IstErkr, MeldeDatum, Refdatum, altersgruppe):
    if (IstErkr == 1):
        # these entries are already ok
        return Refdatum
    
    else:
        # Depending on Altersgruppe subtract Mean of ErkrankungsDatum of MeldeDatum
        if altersgruppe == 'A00-A04':
            delta = M00
        elif altersgruppe == 'A05-A14':
            delta = M05
        elif altersgruppe == 'A15-A34':
            delta = M15
        elif altersgruppe == 'A35-A59':
            delta = M35
        elif altersgruppe == 'A60-A79':
            delta = M60
        elif altersgruppe == 'A80+':
            delta = M80
        else:
            print(Refdatum, " - unknown Altersgruppe found: ", altersgruppe)
            return Refdatum
        
        imputedDate = MeldeDatum - np.timedelta64(delta, 'D')
        return imputedDate

corona['Refdatum'] = corona.apply(lambda x: addMeanErkrankungsDatum(
                    x['IstErkrankungsbeginn'], x['Meldedatum'], x['Refdatum'], x['Altersgruppe']), axis=1)
# Remove cols Altersgruppe, Meldedatum and IstErkrankungsbeginn (we don't need them anymore)
# remaining columns can be seen in the listing of coronaDrop.info()
coronaDrop = corona.drop(['Altersgruppe', 'Meldedatum', 'IstErkrankungsbeginn'], axis=1)
coronaDrop.info()
def sumKreisErkrankungsdatum(X):
    n = X.shape[0]    
    IdLandkreisCol = 4
    RefDatumCol = 5

    # define a list which will hold the new rows collected
    erkrankte = []

    row = X[0]
    lastRefDatum = row[RefDatumCol] 
    lastId = row[IdLandkreisCol]
    
    for iter in range(1,n):
        if (lastRefDatum == X[iter][RefDatumCol]) and (lastId == X[iter][IdLandkreisCol]):
            # same RefDatum and IdLandkreis row read
            # summarize cols AnzahlFall, AnzahlTodesfall, AnzahlGenesen
            for r in [2, 3, 6]:
                row[r] = row[r] + X[iter][r]

        else:
            # another RefDatum and/or IdLandkreis row read
            erkrankte.append(row)
            
            # use current row as the base row
            row = X[iter]
            lastRefDatum = X[iter][RefDatumCol] 
            lastId = X[iter][IdLandkreisCol]

    # append the last row collected in the for-loop that is terminated
    erkrankte.append(row)

    # return a stack from the list erkrankte, a new nparray
    return np.stack(erkrankte, axis=0)

# Note: before performing the calculation sort the whole dataset according 'IdLandkreis' and 'Refdatum'
coronaDrop.sort_values(by=['IdLandkreis', 'Refdatum'], inplace=True)

# Note: if date-time/timezone conversion errors happen then uncomment the last line of the following block
# remove Timezone infos for later processing (once RKI's data contained UTC infos that Pandas could not handle directly)
#print("Datetype of Refdatum:", coronaDrop['Refdatum'].dtype)
#coronaDrop['Refdatum'] = coronaDrop['Refdatum'].dt.tz_localize(None)


# now call the defined function above returning a new nparray
rkiSum = sumKreisErkrankungsdatum(coronaDrop.to_numpy())
rkiSum.shape
# rkiSum[0:9]
# Swapping some columns in rkiSum
# first swap IdKreis and new infections (AnzahlFall)
col1 = 2
col2 = 4
rkiSum.T[[col1, col2]] = rkiSum.T[[col2, col1]]

# second swap RefDatum and Tote
col1 = 3
col2 = 5
rkiSum.T[[col1, col2]] = rkiSum.T[[col2, col1]]
# definition of two helper functions used by 
# computeDoublingFactorAndRnumber and later by createBundeslandProgress

def getHighestMinimum(descList, factor):
    """
    from a list containing descending items get that index which fulfills:
    h = descList[0] (= biggest element of desclist)
    find index i that descList[i] <= h/factor and descList[i-1] > h/factor
    """

    if len(descList) == 0:
        return 0
    
    i = 0
    quot = int(descList[0] / factor)
    for el in descList:
        if el <= quot:
            return i
        else:
            i += 1

    return 1    # default

def calcSmoothedValue(descList, interval):
    """
    get a smoothed (average) value of current NeuInfektionen from a dayrange (interval days)
    This comes close to RKI's definition of Nowcasting with Generationenzeit = 4 (or 7) days
    descList - list containing the total sum of new infections in descending order
    """
    if len(descList) <= interval:
        return descList[0] / len(descList)
    else:
        # descList is somewhat tricky because first element is also the sum of the rest of the list elements
        # We only need to care about its first and its (interval+1)-th element (list starts with index 0)
        return (descList[0] - descList[interval]) / interval
    
def computeDoublingFactorAndRnumber(X, interval):
    """
    Every row in X contains exactly one combination of 'IdLandkreis' and 'RefDatum'
    """
    n = X.shape[0]
    
    # we will add six new columns to X
    X = np.append(X, np.zeros((n,6), dtype=int), axis=1)

    IdLandkreisCol = 2
    AnzahlFallCol = 4
    AnzahlToteCol = 5
    AnzahlGeneseneCol = 6
    
    # new columns to be used
    SumNeuInfCol = 7
    SumToteCol = 8
    SumGenesenCol = 9    
    VerdopplungsCol = 10
    NowcastingCol = 11
    ReprozahlCol = 12
    
    lastKreisId = -1
    sumNeuInf = 0
    sumTote   = 0
    sumGenesene = 0
    
    # collect the cummulated infection numbers
    seqNeuInf = []

    print("computeDoublingFactorAndRnumber: day-interval for Nowcasting =", interval)

    for iter in range(n):
        if lastKreisId == X[iter][IdLandkreisCol]:
            # same Landkreis / Stadt - series
            sumNeuInf += X[iter][AnzahlFallCol]
            X[iter][SumNeuInfCol] = sumNeuInf
            seqNeuInf.insert(0, sumNeuInf)
            
            sumTote += X[iter][AnzahlToteCol]
            X[iter][SumToteCol] = sumTote

            sumGenesene += X[iter][AnzahlGeneseneCol]
            X[iter][SumGenesenCol] = sumGenesene
            
            # VerdopplungsZahl: search backwards in seqNeuInf until 
            #  finding an element e with: e <= sumNeuInf/2.
            #  Verd.zahl = number of indices you stepped back
            X[iter][VerdopplungsCol] = getHighestMinimum(seqNeuInf, 2)
            
            X[iter][NowcastingCol] = calcSmoothedValue(seqNeuInf, interval)
            if X[iter][NowcastingCol] == 0:
                # Normally this case should not happen! When its the case then something is wrong with the RKI
                # data i.e. some data is missing or many corrections (negative new infections) were delivered!
                print ("Warning: Nowcasting <= 0!", X[iter])
                # Choose a value near zero to avoid a Division by Zero Error later in else part
                X[iter][NowcastingCol] = 0.1
            if len(seqNeuInf) < (interval):
                X[iter][ReprozahlCol] = 2.5 # = R_0, accord. RKI the initial value for R (2 - 3)
            else:
                # compute R as a quotient of two interval-day series (Nowcasting)
                X[iter][ReprozahlCol] = X[iter][NowcastingCol] / X[iter-interval+1][NowcastingCol]

        else:
            # another IdLandkreis series will start
            lastKreisId = X[iter][IdLandkreisCol]
            sumNeuInf = X[iter][AnzahlFallCol]
            sumTote   = X[iter][AnzahlToteCol]
            sumGenesene = X[iter][AnzahlGeneseneCol]

            # append new cols cummulated sum of Fälle, Tote, Genesene
            X[iter][SumNeuInfCol] = sumNeuInf
            X[iter][SumToteCol] = sumTote
            X[iter][SumGenesenCol] = sumGenesene
    
            # and additional new cols for the factors and numbers
            seqNeuInf = [sumNeuInf]
            X[iter][VerdopplungsCol] = 1
            X[iter][NowcastingCol] = float(sumNeuInf)
            X[iter][ReprozahlCol]  = 2.5 # approx. the beginning of R for such kind of virus = R_0

    return X

rkiVRfactors = computeDoublingFactorAndRnumber(rkiSum, interval)
# To understand the computation of NewCasting (second last element) and ReprodZahl (last element)
# see next print containing the first cases of SK Flensburg of the resulting npArray rkiVRfactors; 
# the number of new infections is the element after the Timestamp
rkiVRfactors[0:14]
# Original taken from the german 'Statistisches Bundesamt':
# https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/Administrativ/04-kreise.xlsx
#
# For several reasons some data had to be corrected i.e.:
## - shrinking original header to only one header line
## - removing blanks within bigger numbers and replacing ',' by '.' within numbers
## - corrections in the columns 'Regionale Bezeichnung' and 'Kreisfreie Stadt - Landkreis'
## - removing ',' within composed district or town names
## - completing population numbers for the rows containing the name of a state (Bundesland)
## - adding population numbers for 12 districts (IdKreis 11001-11012) of Berlin (numbers taken from RKI Dashboard:
##     https://experience.arcgis.com/experience/478220a4c454480e823b17327b2bf1d4 )
## - completing first row for whole Germany (key (Schlüsselnummer) = 0)
#
# Therefore use this modified file instead the original one, or you must perform all steps by your own ;-)

from os import environ
if environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com':
    # we are in Kaggle environment and expect the url above
    filename = '../input/districts-and-towns-in-germany2019/Districts and Towns in Germany.2019.csv' 
else:
    # we are in another environment, maybe on a local machine
    filename = '/Users/ralfschukey/Documents/Politik/Statistisches_Bundesamt/Districts and Towns in Germany.2019.csv'

landkreise = pd.read_csv(filename, index_col=0, dtype = {'Schlüsselnummer' : 'int64'},
                        usecols=lambda column : column not in ["NUTS3"])
# landkreise.info()
landkreise.head(30)
# add two new columns NeuInfekt100000, Todesfälle100000

def addPopulation100000(X):
    n = X.shape[0]
    X = np.append(X, np.zeros((n,2)), axis=1)

    IdLandkreisCol = 2
    SumInfCol = 7
    SumToteCol = 8

    Inf100000Col = 13
    Tote100000Col = 14

    for iter in range(n):
        bev = landkreise.at[X[iter][IdLandkreisCol], 'Bevölkerung']
#           print(IdLandkreis, bev)
        if (bev == 0):
            X[iter][Inf100000Col] = 0
            X[iter][Tote100000Col] = 0
        else:
            X[iter][Inf100000Col] = (X[iter][SumInfCol] * 100000)/bev
            X[iter][Tote100000Col] = (X[iter][SumToteCol] * 100000)/bev

    return X

rki100000 = addPopulation100000(rkiVRfactors)

# Swap two columns for better grouping
col1 = 10
col2 = 13
rki100000.T[[col1, col2]] = rki100000.T[[col2, col1]]

# rki100000[0:14]
# add two new columns NeueFälle7T and Inzidenz7T (new infected since the last 7 days / per 100000)

def addInzidenz7Days(X, interval):
    n = X.shape[0]
    X = np.append(X, np.zeros((n,2)), axis=1)

    IdLandkreisCol = 2
    ErkrDatumCol = 3
    AnzahlFallCol = 4
    NeueFälle7TCol = 15
    Inzidenz7TCol = 16
    
    lastKreisId = -1
    indexList = []

    for iter in range(n):
        if lastKreisId == X[iter][IdLandkreisCol]:
            # same Landkreis / Stadt - series
            indexList.append(iter)
            
            # add current new case to sumNeuInf
            sumNeuInf += X[iter][AnzahlFallCol]

            currentErkrDatum = X[iter][ErkrDatumCol]
            timeDelta = currentErkrDatum - firstErkrDatum
            
            # current row is more than 7 (interval+1) days away resp. FirstErkrDatum
            while timeDelta.days >= interval:
                # pop first element of list and read the new first element aka index
                firstIndex = indexList.pop(0)
                nextIndex = indexList[0]
                
                # subtract first remembered new case from sumNeuInf
                sumNeuInf -= X[firstIndex][AnzahlFallCol]
                
                firstErkrDatum = X[nextIndex][ErkrDatumCol]
                timeDelta = currentErkrDatum - firstErkrDatum
        
        else:
            # another IdLandkreis row series will start: init some variables
            lastKreisId = X[iter][IdLandkreisCol]
            firstErkrDatum = X[iter][ErkrDatumCol]
            indexList = [iter]
            
            sumNeuInf = X[iter][AnzahlFallCol]
            bev = landkreise.at[lastKreisId, 'Bevölkerung']

        # for each row store NeueFälle7T and Inzidenz7T
        X[iter][NeueFälle7TCol] = sumNeuInf
        X[iter][Inzidenz7TCol ] = (sumNeuInf * 100000)/bev

    return X

rkiInzidenz = addInzidenz7Days(rki100000, interval)
# rkiInzidenz[0:14]
# Movinging last column rkiInzidenz two columns backwards
col1 = 15
col2 = 11
rkiInzidenz.T[[col1, col2]] = rkiInzidenz.T[[col2, col1]]
col1 = 16
col2 = 12
rkiInzidenz.T[[col1, col2]] = rkiInzidenz.T[[col2, col1]]
col1 = 13
col2 = 14
rkiInzidenz.T[[col1, col2]] = rkiInzidenz.T[[col2, col1]]
# rkiInzidenz[0:14]
coronaAll = pd.DataFrame(data=rkiInzidenz, 
         columns=['Bundesland', 'Landkreis', 'IdKreis', 'ErkrDatum', 'Neuinf', 'Tote', 'Genesene', 'FälleSum', 'ToteSum', 'GeneseneSum', 'Fälle100k', 'Neuinf7TSum', 'Inzidenz7T', 'Tote100k', 'VerdZahl', 'Nowcasting', 'ReprodZahl'])
coronaAll
def createBundeslandAndKreisSummary(border):
    '''
    return a new DataFrame containing the summary of all states
    extracted from their Landkreise and Staedte
    Also as a side-affect build a second DataFrame landkreiseLast consisting of the last entry per kreis
    '''
    
    # Problem: the values for ReprodZahl and VerdZahl are calculated via mean() of all Landkreise!
    # This might be incorrect because they tend to be too high!
    # However later in createBundeslandProgress the values for R and V will be recalculated correctly.
    
    #  init all counter
    FaelleSum = 0
    GeneseneSum = 0
    ToteSum = 0
    Neuinf7TSum = 0
    avgV = []
    avgR = []
    populationSum = 0
    populationBuLand = 0
    bundeslandId = 0
    landkreiseLast = None
    KeineNeuenFälle = 0

    lastErkrDatum = coronaAll['ErkrDatum'].describe()['last'] - np.timedelta64(0, 'D')
    print("lastErkrDatum at all: ", lastErkrDatum, ", 7-daysBorder = ", border)

    # iterate over the keys (Schlüsselnummern) of landkreise
    for key in landkreise.index:
        if key == 0:
            # first row should contain: Staat, Deutschland 
            bundesland = landkreise.at[key, 'Kreisfreie Stadt – Landkreis']
            summary = pd.DataFrame({'Bundesland': bundesland,
                                    'FälleSum'   :  0,
                                    'GeneseneSum':  0,
                                    'ToteSum'    :  0,
                                    'Neuinf7TSum':  0,
                                    'Inzidenz7T' :  0.,
                                    'Fälle100k'  :  0.,
                                    'Tote100k'   :  0.,
                                    'VerdZahl'   :  0.,
                                    'ReprodZahl' :  0.,
                                    }, index=[0])
            continue
            
        if key < 20:
            # we read a row of a country (Bundesland)

            # We know that with key == 1 the first Bundesland row will be read
            # As until now nothing is collected skip next if-block

            if key > 1:
                if populationBuLand == 0:
                    print('Error: Population in Bundesland ' + bundesland + ' is zero!')
                    return None
            
                if populationBuLand != populationSum:
                    print('Warning: Found different population numbers in Bundesland ' + bundesland + ':')
                    print('         whole population read', populationBuLand, 'unequals sum of its districts', populationSum)
                
                if populationSum > 0:
                    # post processing of the counters we have read so far
                    land = pd.DataFrame({'Bundesland': bundesland,
                                   'FälleSum'    : FaelleSum,
                                   'GeneseneSum' : GeneseneSum,
                                   'ToteSum'     : ToteSum,
                                   'Neuinf7TSum' : Neuinf7TSum,
                                   'Inzidenz7T'  : Neuinf7TSum * 100000/populationBuLand,
                                   'Fälle100k'   : FaelleSum   * 100000/populationBuLand,
                                   'Tote100k'    : ToteSum     * 100000/populationBuLand,
                                   'VerdZahl'    : pd.Series(avgV, dtype='float16').mean(),
                                   'ReprodZahl'  : pd.Series(avgR, dtype='float16').mean(),
                                  }, index=[bundeslandId])
                    summary = summary.append(land)
                
            # next Bundesland series is starting
            bundeslandId = key
            bundesland = landkreise.at[key, 'Kreisfreie Stadt – Landkreis']
            populationBuLand = int(landkreise.at[key, 'Bevölkerung'])
            print('Processing districts and towns for', landkreise.at[key, 'Regionale Bezeichnung'], bundesland, '...')
                
            #  init all counter
            populationSum = 0
            FaelleSum = 0
            GeneseneSum = 0
            ToteSum = 0
            Neuinf7TSum = 0
            avgV = []
            avgR = []

        else:
            if key < 1000:
                # got row for a region of a Bundesland, ignore it
                continue
                
            # we are within a series of one Bundesland
            
            # 1. interpolate the next last ErkrDatum of the actual landkreisID
            coronaKreis = coronaAll[coronaAll['IdKreis'] == key]['ErkrDatum']
            if (coronaKreis.size == 0):
                # There is no data for this district available
                print("Warning: No data found for IdKreis", landkreise.at[key, 'Kreisfreie Stadt – Landkreis'])
                continue
                
            foundIndex = coronaKreis.tail(1).index
            ser = coronaAll.iloc[foundIndex]

            # 2. sum / add all counters
            populationSum = populationSum + int(landkreise.at[key, 'Bevölkerung'])

            FaelleSum = FaelleSum + int(ser['FälleSum'])
            GeneseneSum = GeneseneSum + int(ser['GeneseneSum'])
            ToteSum = ToteSum + int(ser['ToteSum'])
            avgV.append(float(ser['VerdZahl']))
            avgR.append(float(ser['ReprodZahl']))
            
            timeDelta = lastErkrDatum - ser['ErkrDatum'].values[0]
            if timeDelta.days < border:
                Neuinf7TSum = Neuinf7TSum + int(ser['Neuinf7TSum'])
                if landkreiseLast is None:
                    landkreiseLast = ser
                else:
                    landkreiseLast = landkreiseLast.append(ser)

            else:
                KeineNeuenFälle += 1
                # print out the kreis/town that had currently no new infections
#                print(key, ": Last Erkr.Datum of ", landkreise.at[key, 'Kreisfreie Stadt – Landkreis'],
#                      "is too far:", timeDelta.days, "days")

    if populationBuLand == 0:
        print('Error: Population in Bundesland ' + bundesland + ' is zero!')
        return None
            
    if populationSum > 0:
        # for the last Bundesland collected store its values
        df = pd.DataFrame({'Bundesland'  : bundesland,
                       'FälleSum'    : FaelleSum,
                       'GeneseneSum' : GeneseneSum,
                       'ToteSum'     : ToteSum,
                       'Neuinf7TSum' : Neuinf7TSum,
                       'Inzidenz7T'  : Neuinf7TSum * 100000/populationSum,
                       'Fälle100k'   : FaelleSum   * 100000/populationSum,
                       'Tote100k'    : ToteSum     * 100000/populationSum,
                       'VerdZahl'    : pd.Series(avgV, dtype='float16').mean(),
                       'ReprodZahl'  : pd.Series(avgR, dtype='float16').mean(),
                       }, index=[bundeslandId])
        summary = summary.append(df)
    
    print("Number of districts (total sum = 412) with no recent cases:", KeineNeuenFälle)
    return summary, landkreiseLast

# carefully adjust dateBorder!
daysBorder = 8 + round(max(M00, M05, M15, M35, M60, M80) / 2)  # 
bundeslaenderSum, landkreiseLast = createBundeslandAndKreisSummary(daysBorder)

# in landkreiseLast use 'IdKreis' as index key because it is ambiguous now
landkreiseLast.set_index('IdKreis', inplace=True)
landkreiseLast.describe()
def fillDeutschlandrow():

    # summarize and average (mean) columnwise
    srSum  = bundeslaenderSum.sum(axis=0)
    srMean = bundeslaenderSum.mean(axis=0)

    # first row 'Deutschland' Insgesamt
    bevDeutschland = int(landkreise.at[0, 'Bevölkerung'])
    bundeslaenderSum.at[0, 'FälleSum'   ] = srSum['FälleSum']
    bundeslaenderSum.at[0, 'ToteSum'    ] = srSum['ToteSum']
    bundeslaenderSum.at[0, 'GeneseneSum'] = srSum['GeneseneSum']
    bundeslaenderSum.at[0, 'Neuinf7TSum'] = srSum['Neuinf7TSum']
    bundeslaenderSum.at[0, 'Inzidenz7T' ] = srMean['Inzidenz7T']
    bundeslaenderSum.at[0, 'Fälle100k'  ] = (srSum['FälleSum'] * 100000)/bevDeutschland
    bundeslaenderSum.at[0, 'Tote100k'   ] = (srSum['ToteSum'] * 100000)/bevDeutschland
    bundeslaenderSum.at[0, 'VerdZahl'   ] = srMean['VerdZahl']
    bundeslaenderSum.at[0, 'ReprodZahl' ] = srMean['ReprodZahl']

fillDeutschlandrow()
bundeslaenderSum
# Calculate deathrate (Todesrate) per 100000 in bundeslaenderSum
def calculateDeathRate(Fälle, Tote):
    return 100 * Tote / Fälle

def swapDFTwoColumns(df, col1, col2):
    # swap new (last) column with VerdZahl
    cols = list(df.columns)
    cols[col1], cols[col2] = cols[col2], cols[col1]
    return df[cols]

bundeslaenderSum['Todesrate100k'] = bundeslaenderSum.apply(
                lambda x: calculateDeathRate(x['Fälle100k'], x['Tote100k']), axis=1)

bundeslaenderSum = swapDFTwoColumns(bundeslaenderSum, 8, 10)
#  sort bundeslaenderSum by Bundesland lexicographically, incl. the whole entry for Germany (Deutschland)
bundeslaenderSum.sort_values(by=['Bundesland'], ascending=True)
bundeslaenderOnly = bundeslaenderSum.drop(index=0)
bundeslaenderOnly.sort_values(by=['Bundesland'], ascending=True, inplace=True)
bundeslaenderOnly
# Plot some Pie Charts concerning BundeslaenderOnly (sorted by Bundesland name)
total = bundeslaenderOnly['Fälle100k'].sum()
_ = bundeslaenderOnly.Fälle100k.plot.pie(labels=bundeslaenderOnly['Bundesland'].values, pctdistance=0.8,
            fontsize=14, figsize=(12,12), startangle=90, autopct=lambda p: '{:.0f}'.format(p * total/100))
# Plot some Pie Charts concerning BundeslaenderOnly (sorted by Bundesland name)
total = bundeslaenderOnly['Inzidenz7T'].sum()
_ = bundeslaenderOnly.Inzidenz7T.plot.pie(labels=bundeslaenderOnly['Bundesland'].values, pctdistance=0.8,
            fontsize=14, figsize=(12,12), startangle=90, autopct=lambda p: '{:.2f}'.format(p * total/100))
# 1. list of all districts/towns that seem to be / contain a Corona hotspot (sort by Fälle100k, alt. Tote100k)
hotspots1 = landkreiseLast.sort_values(by=['Fälle100k'], ascending=False).head(25)

# print a list of districts (Landkreise) having the 'highest score' considering the selected code (here: Fälle100k)
hotspots1
total1 = hotspots1['Fälle100k'].sum()
_ = hotspots1.Fälle100k.plot.pie(labels=hotspots1['Landkreis'].values, pctdistance=0.85,
            fontsize=14, figsize=(12,12), startangle=0, autopct=lambda p: '{:.2f}'.format(p * total1/100))
# 2. list of all districts/towns that seem to be or contain a COVID-19 hotspot (sort by Inzidenz7T)
hotspots2 = landkreiseLast.sort_values(by=['Inzidenz7T'], ascending=False).head(25)

# print a list of districts (Landkreise) having the 'highest score' considering the selected code (here: Inzidenz7T)
hotspots2
total2 = hotspots2['Inzidenz7T'].sum()
_ = hotspots2.Inzidenz7T.plot.pie(labels=hotspots2['Landkreis'].values, pctdistance=0.9,
            fontsize=14, figsize=(12,12), startangle=0, autopct=lambda p: '{:.2f}'.format(p * total2/100))
coronaDistrict = coronaAll[coronaAll['Landkreis'] == hotspots2.iat[0, 1]]     # hotspots2.iat[0, 1]

# now we can change the index to 'Erkr.Datum' because it is unambiguously
coronaDistrict.set_index('ErkrDatum', inplace=True)
coronaDistrict.tail(25)
# plot two parameters, both use a linear legend
# Depending on the course of 'Neuinf' it makes sense to use a log scale (try it out)
landkreis = coronaDistrict.at[coronaDistrict.index[0], 'Landkreis']
title = 'Course of COVID-19 in ' + landkreis

fig1, ax1 = plt.subplots(figsize=(18,10))
ax1.set_title(title)
coronaDistrict.Neuinf.plot(legend=True, logy=False)
_ = coronaDistrict.VerdZahl.plot(secondary_y=True, legend=True)
# plot two other parameters, both use a log scale
fig2, ax2 = plt.subplots(figsize=(18,10))
ax2.set_title(title)
coronaDistrict.Inzidenz7T.plot(legend=True, logy=True)
_ = coronaDistrict.ReprodZahl.plot(secondary_y=True, logy=True, legend=True)
# look at the Hotspots (resp. Fälle100k or Tote100k) of a specific state (Bundesland)
specificBundesland = 'Nordrhein-Westfalen'    # Nordrhein-Westfalen
Bundesland = landkreiseLast[landkreiseLast['Bundesland'] == specificBundesland]

# sort by the first code
BundeslandHead1 = Bundesland.sort_values(by=['Tote100k'], ascending=False).head(25)
BundeslandHead1
total = BundeslandHead1['Tote100k'].sum()
_ = BundeslandHead1.Tote100k.plot.pie(labels=BundeslandHead1['Landkreis'].values, pctdistance=0.9,
            fontsize=12, figsize=(12,12), startangle=0, autopct=lambda p: '{:.2f}'.format(p * total/100))
# sort by the second code e.g. Inzidenz7T
BundeslandHead2 = Bundesland.sort_values(by=['Inzidenz7T'], ascending=False).head(25)
BundeslandHead2
total = BundeslandHead2['Inzidenz7T'].sum()
_ = BundeslandHead2.Inzidenz7T.plot.pie(labels=BundeslandHead2['Landkreis'].values, pctdistance=0.9,
            fontsize=12, figsize=(12,12), startangle=0, autopct=lambda p: '{:.2f}'.format(p * total/100))
coronaErkrBegin = coronaAll.sort_values(by=['ErkrDatum', 'IdKreis'], ascending=True, inplace=False)
coronaErkrBegin.describe()
# (re-)calculate Neuinf7T and Inzidenz7T in DataFrame bLand
# bLand's index consists of Erkr.Datum timestamps which is the basis for calculation
# this function will be used within createBundeslandProgress (s. next)

def calcInzidenz7Days(bLand, population, interval):
    # make a copy of the input DataFrame
    bLandNew = bLand.copy(deep=True)
    
    sumNeuInf = 0
    
    # indexList holds the last dates
    indexList = []

    for currentErkrDatum in bLandNew.index:
        if indexList == []:
            # mark beginning of a list
            firstErkrDatum = currentErkrDatum
            sumNeuInf = bLandNew.at[currentErkrDatum, 'FälleNeu']
            indexList = [currentErkrDatum]
        else:
            sumNeuInf += bLandNew.at[currentErkrDatum, 'FälleNeu']
            indexList.append(currentErkrDatum)
            timeDelta = currentErkrDatum - firstErkrDatum
            
            # current row is more than interval days away resp. FirstErkrDatum
            while timeDelta.days >= interval:
                # remove first element of list and read the new first element aka Erkr.Datum
                removeErkrDatum = indexList.pop(0)
                sumNeuInf -= bLandNew.at[removeErkrDatum, 'FälleNeu']
                firstErkrDatum = currentErkrDatum if indexList == [] else indexList[0]                    
                timeDelta = currentErkrDatum - firstErkrDatum
            
        # for each row store NeueFälle7T and Inzidenz7T
        bLandNew.at[currentErkrDatum, 'Neuinf7TSum'] = sumNeuInf
        bLandNew.at[currentErkrDatum, 'Inzidenz7T' ] = sumNeuInf * 100000/population

    return bLandNew
# Starting from first to last known Erkr.Datum in coronaErkrBegin for each timestamp collect
# - for every bundesland the numbers from their landkreise ==> 16 DataFrames
# - for whole Germany the numbers from their bundeslaender ==>  1 DataFrame

def createBundeslandProgress(IdBundesland, interval):
    '''
    return a new DataFrame containing the progress of all numbers of a certain Bundesland
    as data use coronaErkrDatum, sorted by ErkrDatum und IdKreis and landkreise
    TODO: Neuinf7TSum and Inzidenz7T have to be recalculated afterwards
    '''
    
    #  init all counter
    FaelleNeu = -1
    ToteNeu = 0
    GeneseneNeu = 0    
    FaelleSum = 0
    ToteSum = 0
    GeneseneSum = 0
    
    # here we will store all collected entries
    erkrDatumList = None
    seqNowcasting = []

    bundesland = landkreise.at[IdBundesland, 'Kreisfreie Stadt – Landkreis']

    if IdBundesland == 0:
        # iterate thru all states
        coronaLand = coronaErkrBegin
    else:
        coronaLand = coronaErkrBegin[coronaErkrBegin['Bundesland'] == bundesland]

    # test the Parameter
    if len(coronaLand) == 0:
        print("Could not find entries for the state", bundesland)
        return None
    
    populationBuLand = int(landkreise.at[IdBundesland, 'Bevölkerung'])
    lastErkrDatum = pd.to_datetime('2019-12-12')  # init with a dummy date

    # for each ErkrDatum found summarize all items of coronaLand with this date in:
    for iter in coronaLand.index:
        currentErkrDatum = coronaLand.at[iter, 'ErkrDatum']
        timeDelta = currentErkrDatum - lastErkrDatum
        
        if timeDelta.days > 0:
            # read row with a newer ErkrDatum ==> save current erkrDay
            if FaelleNeu == -1:
                # first time when entering the loop, read value before loop begin
                FaelleNeu   = coronaLand.at[iter, 'Neuinf']
                ToteNeu     = coronaLand.at[iter, 'Tote']
                GeneseneNeu = coronaLand.at[iter, 'Genesene']
                seqNeuInf   = []
                seqNowcasting = [float(FaelleNeu)]
                lastErkrDatum = currentErkrDatum
                continue
                
            else:
                # we already read the next row but before save the collected so far
                FaelleSum   += FaelleNeu
                ToteSum     += ToteNeu
                GeneseneSum += GeneseneNeu
                seqNeuInf.insert(0, FaelleSum)

            # Nowcasting, Reproduktionszahl: use a list for storing the last interval nowcasting entries
            currentNow = calcSmoothedValue(seqNeuInf, interval)
            seqNowcasting.append(currentNow)
            
            if len(seqNowcasting) <= interval:
                reprozahl = 2.5 # = R_0
            else:
                # compute a smoothed R as a Quotient of two interval-day series (Nowcasting)
                nowCastingFirst = seqNowcasting.pop(0)
                reprozahl = currentNow / nowCastingFirst                

            erkrDay = pd.DataFrame({'Bundesland' : bundesland,
                            'FälleNeu'   : FaelleNeu,
                            'ToteNeu'    : ToteNeu,
                            'GeneseneNeu': GeneseneNeu,
                            'FälleSum'   : FaelleSum,
                            'ToteSum'    : ToteSum,
                            'GeneseneSum': GeneseneSum,
                            'Fälle100k'  : FaelleSum * 100000 / populationBuLand,
                            'Neuinf7TSum': 0,
                            'Inzidenz7T' : 0.,
                            'Tote100k'   : ToteSum * 100000 / populationBuLand,
                            'VerdZahl'   : getHighestMinimum(seqNeuInf, 2),
                            'Nowcasting' : currentNow,
                            'ReprodZahl' : reprozahl
                           }, index=[lastErkrDatum])

            lastErkrDatum = currentErkrDatum

            # store the collected items so far
            if erkrDatumList is None:
                erkrDatumList = erkrDay
            else:
                erkrDatumList = erkrDatumList.append(erkrDay)
                
            #  init all counter
            FaelleNeu   = coronaLand.at[iter, 'Neuinf']
            ToteNeu     = coronaLand.at[iter, 'Tote']
            GeneseneNeu = coronaLand.at[iter, 'Genesene']
            
        else:
            # read another row with same ErkrDatum; sum / add all counters
            FaelleNeu   += coronaLand.at[iter, 'Neuinf']
            ToteNeu     += coronaLand.at[iter, 'Tote']
            GeneseneNeu += coronaLand.at[iter, 'Genesene']
            
    # for the last entry collected store its values
    FaelleSum   += FaelleNeu
    ToteSum     += ToteNeu
    GeneseneSum += GeneseneNeu

    nowCastingFirst = seqNowcasting.pop(0)
    reprozahl = currentNow / nowCastingFirst                

    erkrDay = pd.DataFrame({'Bundesland' : bundesland,
                            'FälleNeu'   : FaelleNeu,
                            'ToteNeu'    : ToteNeu,
                            'GeneseneNeu': GeneseneNeu,
                            'FälleSum'   : FaelleSum,
                            'ToteSum'    : ToteSum,
                            'GeneseneSum': GeneseneSum,
                            'Fälle100k'  : FaelleSum * 100000 / populationBuLand,
                            'Neuinf7TSum': 0,
                            'Inzidenz7T' : 0.,
                            'Tote100k'   : ToteSum * 100000 / populationBuLand,
                            'VerdZahl'   : getHighestMinimum(seqNeuInf, 2),
                            'Nowcasting' : currentNow,
                            'ReprodZahl' : reprozahl
                           }, index=[currentErkrDatum])
            
    # store the collected items so far
    erkrDatumList = erkrDatumList.append(erkrDay)
    
    erkrListInzidenz = calcInzidenz7Days(erkrDatumList, populationBuLand, interval)
    return erkrListInzidenz

# In a loop (range(1,16)) you could generate a DataFrame for every state (Bundesland)
# Instead as an example we pick one or two of them
# example1: Id = 5, North Rhine-Westphalia (Nordrhein Westfalen)
coronaBundesland1 = createBundeslandProgress(5, interval)

if coronaBundesland1 is not None:
    coronaBundesland1['Todesrate100k'] = coronaBundesland1.apply(
                lambda x: calculateDeathRate(x['Fälle100k'], x['Tote100k']), axis=1)

    # swap new column Todesrate100k with VerdZahl
    coronaBundesland1 = swapDFTwoColumns(coronaBundesland1, 11, 14)

coronaBundesland1
# example2: Id = 13, Mecklenburg-Western Pomerania (Mecklenburg-Vorpommern)
coronaBundesland2 = createBundeslandProgress(13, interval)

if coronaBundesland2 is not None:
    coronaBundesland2['Todesrate100k'] = coronaBundesland2.apply(
                lambda x: calculateDeathRate(x['Fälle100k'], x['Tote100k']), axis=1)

    # swap new column Todesrate100k with VerdZahl
    coronaBundesland2 = swapDFTwoColumns(coronaBundesland2, 11, 14)
    
coronaBundesland2
# plot two parameters of coronaBundesland1, here both y-axes use a logarithmic scale
def plotCoronaState(coronaState):
    name = coronaState.at[coronaState.index[0], 'Bundesland']

    title = 'Course of COVID-19 data I in state ' + name
    _, ax1 = plt.subplots(figsize=(18,10))
    ax1.set_title(title)
    coronaState.FälleNeu.plot(legend=True)
    coronaState.VerdZahl.plot(secondary_y=True, legend=True)
    
    title = 'Course of COVID-19 data II in state ' + name
    _, ax2 = plt.subplots(figsize=(18,10))    
    ax2.set_title(title)
    coronaState.Inzidenz7T.plot(legend=True, logy=True)
    coronaState.ReprodZahl.plot(secondary_y=True, legend=True, logy=True)

# for both example states call the function
if coronaBundesland1 is not None: plotCoronaState(coronaBundesland1)
if coronaBundesland2 is not None: plotCoronaState(coronaBundesland2)
# example1: Id = 5, North Rhine-Westphalia (Nordrhein Westfalen)
coronaGermany = createBundeslandProgress(0, interval)

coronaGermany['Todesrate100k'] = coronaGermany.apply(
              lambda x: calculateDeathRate(x['Fälle100k'], x['Tote100k']), axis=1)

# swap new column Todesrate100k with VerdZahl
coronaGermany = swapDFTwoColumns(coronaGermany, 11, 14)

coronaGermany.tail(25)
plotCoronaState(coronaGermany)
# Pick up a particular district with all its rows   
coronaDistrict2 = coronaAll[coronaAll['Landkreis'] == 'LK Gütersloh']

# now we can change the index to 'Erkr.Datum' because it is unambiguously
coronaDistrict2.set_index('ErkrDatum', inplace=True)
coronaDistrict2.tail(25)
# plot two parameters; Attention: in this example Neuinf uses a log scale
landkreis = coronaDistrict2.at[coronaDistrict2.index[0], 'Landkreis']

title = 'Course of COVID-19 data I in ' + landkreis
fig1, ax1 = plt.subplots(figsize=(18,10))
ax1.set_title(title)
coronaDistrict2.Neuinf.plot(legend=True, logy=True)
_ = coronaDistrict2.VerdZahl.plot(secondary_y=True, legend=True)
# plot two other parameters; Attention: in this example Inzidenz7T uses a log scale
title = 'Course of COVID-19 data II in ' + landkreis
fig2, ax2 = plt.subplots(figsize=(18,10))
ax2.set_title(title)
coronaDistrict2.Inzidenz7T.plot(legend=True, logy=True)
_ = coronaDistrict2.ReprodZahl.plot(secondary_y=True, logy=False, legend=True)