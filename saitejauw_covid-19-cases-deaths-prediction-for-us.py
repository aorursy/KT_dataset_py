# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from datetime import datetime, timedelta
from ipywidgets import interact
import itertools
from tabulate import tabulate


confirmedFilename = '/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv'
deathsFilename = '/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv'
recoveredFilename = '/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_recovered_global.csv'

# Countries to ski prediction
skipPredictionCountriesList = ['China', 'Korea, South', 'Singapore', 'Taiwan*']
# Load all 3 csv files
covidFrDict = {}
covidFrDict['confirmed'] = pd.read_csv(confirmedFilename)
covidFrDict['deaths'] = pd.read_csv(deathsFilename)
covidFrDict['recovered'] = pd.read_csv(recoveredFilename)
# Get list of dates
colNamesList = list(covidFrDict['confirmed'])
dateList = [colName for colName in colNamesList if '/20' in colName] # Dates always have '/20' in them

# Create list of datetime objects from date strings
dateTimeOjectList = [datetime.strptime(timeStr, '%m/%d/%y') for timeStr in dateList]
# Function to get all three frames for a given country
def getCountryCovidFrDict(countryName):
    countryCovidFrDict = {}
    for key in covidFrDict.keys():
        dataFr = covidFrDict[key]
        countryCovidFrDict[key] = dataFr[dataFr['Country/Region'] == countryName]
    return countryCovidFrDict

# Function for plotting country data
def plotCountryData(countryName, logScale=False):
    countryCovidFrDict = getCountryCovidFrDict(countryName)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    for key in countryCovidFrDict.keys():
        casesByDateDict = dict(countryCovidFrDict[key][dateList].sum(axis=0))
        # Stop drawing vertical lines on log scale when plotting zero
        if logScale:
            for dateKey in casesByDateDict.keys():
                if casesByDateDict[dateKey] == 0:
                    casesByDateDict[dateKey] = np.nan
        ax.plot(list(casesByDateDict.keys()), list(casesByDateDict.values()), marker='o', label=key);

    plt.xticks(rotation=45, ha="right");
    
    if logScale:
        plt.yscale('log')

    every_nth = 4
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    ax.legend(loc='upper left');
    plt.title('Data for {}'.format(countryName), fontsize=26);
    plt.xlabel('Date', fontsize=18);
    plt.ylabel('Number of cases', fontsize=18);
plotCountryData('US', logScale=False)
def getPredictionsForFuture(countryName,
                            nDays=5,
                            invertible=False, # Whether model is invertible or not
                            plot=True,
                            logScale=False,
                            grid=None,
                            printResults=True,
                            tablesToUse='all',
                            returnResults=False):
   
    
    # Extract model parameters
    p, d, q = (1, 2, 2)
    countryCovidFrDict = getCountryCovidFrDict(countryName)
    plotStartedFlag = False
    if tablesToUse == 'all':
        keysList = countryCovidFrDict.keys()
    else:
        keysList = tablesToUse
    allData = {}
    predData = {}
    for key in keysList:
        if printResults:
            print('Table type:', key)
        data = list(countryCovidFrDict[key][dateList].sum(axis=0))
        predictionsList = []

        for i in range(nDays):
            if invertible:
                model = SARIMAX(data, order=(p, d, q))

                model_fit = model.fit(disp=False)

                # make prediction
                yhat = model_fit.predict(len(data), len(data), typ='levels')
            else:
                model = SARIMAX(data, order=(p, d, q), enforce_invertibility=False)

                model_fit = model.fit(disp=False)

                # make prediction
                yhat = model_fit.predict(len(data), len(data), typ='levels')

            data.extend(yhat)
            predictionsList.append(yhat[0])
            
        # Required for printing as well as plotting
        dateTimeOjectForPlotList = dateTimeOjectList.copy()
        lastDateTimeObject = dateTimeOjectForPlotList[-1]
        futureDateTimeObjectList = []
        for i in range(nDays):
            lastDateTimeObject += timedelta(days=1)
            dateTimeOjectForPlotList.append(lastDateTimeObject)
            futureDateTimeObjectList.append(lastDateTimeObject)

        datetimeForPlotList = [dateTimeObject.strftime('%m/%d/%y') for dateTimeObject in dateTimeOjectForPlotList]
        futureDateTimeList = [dateTimeObject.strftime('%m/%d/%y') for dateTimeObject in futureDateTimeObjectList]
        
        if printResults:
            print('Predictions for next {} days:'.format(nDays))
            # Round off predictions for printing
            predPrintList = [np.around(elem) for elem in predictionsList]
            datePredList = list(zip(futureDateTimeList, predPrintList))
            # Convert individual elements of zip to a list
            datePredList = [list(elem) for elem in datePredList]
            print(tabulate(datePredList, headers=['Date', 'Prediction'], tablefmt='orgtbl'))
            
                
        if plot:
            # Start a plot if not already started
            if plotStartedFlag == False:
                fig = plt.figure(figsize=(8,8))
                ax = fig.add_subplot(111)
                plotStartedFlag = True
            if logScale:
                for i in range(len(data)):
                    if data[i] == 0:
                        data[i] = np.nan
            ax.plot(datetimeForPlotList, data, marker='o', label=key);
            # Circle predictions
            ax.scatter(futureDateTimeList, predictionsList, s=130, linewidth=2, facecolors='none', edgecolors='k');
        
        allDataDict = dict(zip(datetimeForPlotList, data))
        allData[key] = allDataDict

        predDict = dict(zip(futureDateTimeList, predictionsList))
        predData[key] = predDict
    if plot:
        if logScale:
            plt.yscale('log')
        plt.xticks(rotation=45, ha="right");
        
        every_nth = 4
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        ax.legend(loc='upper left', prop={'size': 18});
        plt.title('Data for {}\n(Including predictions for next {} days)'.format(countryName, nDays), fontsize=24);
        plt.xlabel('Date', fontsize=18);
        if logScale:
            plt.ylabel('Number of cases (log scale)', fontsize=18);
        else:
            plt.ylabel('Number of cases', fontsize=18);
        
        if grid != None:
            plt.grid(axis=grid)
        
    if returnResults:
        return allData, predData
allData, predData = getPredictionsForFuture('US', 
                                            invertible=False,
                                            plot=True,
                                            logScale=False, 
                                            printResults=True, 
                                            nDays=5, 
                                            tablesToUse=['confirmed'], 
                                            grid='y',
                                            returnResults=True)
def comparePlotsOfNCountries(countryNameList,
                             nDays=5,
                             invertible=False, # Whether model is invertible or not
                             logScale=False,
                             grid=None,
                             printResults=True,
                             tableToUse='confirmed'):
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    printListCreatedFlag = False
    for countryName in countryNameList:
        allData, predData = getPredictionsForFuture(countryName,
                                                    nDays=nDays,
                                                    invertible=invertible,
                                                    plot=False,
                                                    logScale=logScale,
                                                    grid=grid,
                                                    printResults=False,
                                                    tablesToUse=[tableToUse],
                                                    returnResults=True)
        allData = allData[tableToUse]
        predData = predData[tableToUse]
        ax.plot(list(allData.keys()), list(allData.values()), marker='o', label=countryName);
        
        if printListCreatedFlag == False:
            printListCreatedFlag = True
            
            futureDateTimeList = list(predData.keys())
            predictionsList = list(predData.values())
            predPrintList = [np.around(elem) for elem in predictionsList]
            
            # Zip dates and predictions together
            datePredList = list(zip(futureDateTimeList, predPrintList))
            # Convert individual elements of zip to a list
            datePredList = [list(elem) for elem in datePredList]
        else:
            predictionsList = list(predData.values())
            for i in range(len(datePredList)):
                datePredList[i].append(np.around(predictionsList[i]))
                
        # Circle predictions
        ax.scatter(futureDateTimeList, predictionsList, s=130, linewidth=2, facecolors='none', edgecolors='k');
            
    if printResults:
        headerList = ['Date']
        headerList.extend(countryNameList)
        print(tabulate(datePredList, headers=headerList, tablefmt='orgtbl'))
            
        
    plt.xticks(rotation=45, ha="right");
    if logScale:
        plt.yscale('log')
        
    every_nth = 4
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    ax.legend(loc='upper left', prop={'size': 18});
    
    plt.title('Covid-19 data ({})\nIncluding predictions for next {} days'.format(tableToUse, nDays), fontsize=24);
    plt.xlabel('Date', fontsize=18);
    if logScale:
        plt.ylabel('Number of cases (log scale)', fontsize=18);
    else:
        plt.ylabel('Number of cases', fontsize=18);
        
    if grid != None:
        plt.grid(axis=grid)
comparePlotsOfNCountries(['US', 'Italy', 'Germany', 'Spain', 'China'], nDays=3, grid='y', tableToUse='deaths')