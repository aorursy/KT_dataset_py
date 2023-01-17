# essential libraries

#    Commenting-out libraries I never used

# import json

# import random

# from urllib.request import urlopen



# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib as mpl

import matplotlib.pyplot as pyplot

# Used to silence the warning that too many graphs are being generated.  The default is 100 if I recall correctly

mpl.rc('figure', max_open_warning =1000)

# import seaborn as sns

# import plotly.express as px

# import plotly.graph_objs as go

# import plotly.figure_factory as ff

# import calmap

# import folium



# color pallette -- thanks, Kaggle.

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active case - yellow



# converter

import matplotlib.dates as mdates

# from pandas.plotting import register_matplotlib_converters

# register_matplotlib_converters()   



# hide warnings -- this would be another way to hide warnings.  Thanks again, Kaggle, lol.

import warnings

warnings.filterwarnings('ignore')



# html embedding

# from IPython.display import Javascript

# from IPython.core.display import display

# from IPython.core.display import HTML



# others I used that aren't listed above

import requests

import re

import os

import logging

from datetime import datetime as dt

import itertools as itr
def getFileList(homepage):

    """Use BeautifulSoup to fetch a file list from the GitHub page stored in homepage"""

    source = requests.get(homepage).text

    soup = BeautifulSoup(source, 'lxml')

    files = []

    for markup in soup.find_all('a', class_='js-navigation-open'):

        arr = markup.get_text().split('\n')

        if re.search('\d\d\-\d\d\-\d\d\d\d.csv',arr[0]):

            files.append(arr[0])

    return files



def exDate(sDat, sFormat='%m-%d-%Y'):

    """remove the extension from a file name formated in sFormat and return a datetime object"""

    return dt.strptime(sDat[0:-4], sFormat)



def strfdat(datIn, sFormat = '%m-%d-%Y'): 

    """return a filename with a .csv extension and a date formatted string name"""

    return datIn.strftime(sFormat)+'.csv'



def trimFileList(fetchList, datCutOff, sFormat = '%m-%d-%Y'):

    """Pop any files of the form %m-%d-%Y.csv out of fetchList if they occur on or before any of the files in dirList"""

    fetchFile = fetchList

    retList = []

    for n in range(len(fetchFile)):

        if dt.strptime(fetchFile[n].split('.')[0], sFormat) > datCutOff: 

            retList.append(fetchFile[n])

    return retList



def removeUndated(fileList, sFormat = '%m-%d-%Y'):

    undatedRemoved = fileList

    n = 0

    while n <= len(undatedRemoved):

        try:

            n += 1

            dt.strptime(undatedRemoved[n-1].split('.')[0], sFormat)

        except ValueError:

            pop = undatedRemoved.pop(n-1)

            n -= 1

        except IndexError:

            break

    return undatedRemoved



def sortFileList(fileList):

    fl = np.array(list(map(exDate, fileList)))

    fl = np.sort(fl)

    #Convert sorted arrray back to an array of file names

    fl = list(map(strfdat, fl))

    return fl



def grabDataFromURL(urlroot, filename):

    url = urlroot+'/'+filename

    logger.info('grabDataFromURL() - Grabbing and adding data from file at ' + url)

    #Check the date and grab the data according the the correct schema

    if dt.strptime(filename.split('.')[0], '%m-%d-%Y') < dt.strptime("03-01-2020", '%m-%d-%Y'):

        df2 = pd.read_csv(filepath_or_buffer = url, sep = ',', index_col= False, header = 0,\

                         names = ['province', 'country', 'recorded', 'confirmed', 'deaths', 'recovered'])

    elif dt.strptime(filename.split('.')[0], '%m-%d-%Y') < dt.strptime("03-22-2020", '%m-%d-%Y'):

        df2 = pd.read_csv(filepath_or_buffer = url, sep = ',', index_col= False, header = 0,\

                names = ['province', 'country', 'recorded', 'confirmed', 'deaths', 'recovered', 'lat', 'lon'])

    elif dt.strptime(filename.split('.')[0], '%m-%d-%Y') < dt.strptime("05-29-2020", '%m-%d-%Y'):

        df2 = pd.read_csv(filepath_or_buffer = url, sep = ',', index_col= False, header = 0,\

                names = ['fips', 'city', 'province', 'country', 'recorded', 'lat', 'lon', 'confirmed', 'deaths', 'recovered', 'active', 'combined_key'])        

    else:

        df2 = pd.read_csv(filepath_or_buffer = url, sep = ',', index_col= False, header = 0,\

                names = ['fips', 'city', 'province', 'country', 'recorded', 'lat', 'lon', 'confirmed', 'deaths', 'recovered', 'active', 'combined_key', 'incidence_rate', 'case-fatality_ratio'])

    #log what's been grabbed

    logger.info('grabDataFromURL() - Data grabbed is \n' + str(df2.values) + '\n')

    return df2                    



#define variables

homepage = 'https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports'

urlroot = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports'    

localpath = 'C:\\Users\\Sean\\Documents\\Dev\\Covid-19\\'

    

#Create logger

LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(filename = localpath+'Covid-19.log',

                   level = logging.DEBUG,

                   format = LOG_FORMAT,

                   filemode = 'w')

logger = logging.getLogger()



# Execution

try: 

    fl = getFileList(homepage)

    # Convert fl to to numpy series of dates to prepare for descending sort

    ldir = os.listdir(localpath)

    # Remove the files from ldir that don't conform to a date

    ldir = removeUndated(ldir)    

    ldir = sortFileList(ldir)

    # Remove files from fl that have already been grabbed and sit in local files

    if len(ldir) > 0:

        fl = trimFileList(fl, exDate(ldir[-1]))

    # Sort the remaining files in the file list

    fl = sortFileList(fl)         

    # Prepare empty dataframe for population

    df = pd.DataFrame()

    if len(ldir) > 0:

        df = pd.read_csv(localpath+ldir[-1])        

    for n in fl:

        df2 = grabDataFromURL(urlroot, n)

        df2.insert(0, 'file', n.split('.')[0])

        df = df.append(df2)

    # Drop columns like "Unnamed:*" that are getting created in the Dataframe.append() process

    rexp = re.compile('^Unnamed:')

    for n in list(df.columns):

        if rexp.match(n):

            df.drop(n, axis =1, inplace = True)    

    # save the accumulated file locally

    if len(fl) > 0:

        if fl[-1] not in os.listdir(localpath):

            df.to_csv(localpath+'\\'+fl[-1])        

    logger.info('pull_data() - output \n'+str(df.values))

    

finally:

    logging.shutdown()
def pull_data(url):

    df = pd.read_csv(url)

    return df



def add_column(sName, df, fFunc):

    df2 = df.copy()

    newCol = []

    for n in range(0, df.shape[0]):

        newCol.append(fFunc(df2, n))

    df2[sName] = newCol

    return df2    

    

def get_mDate(df, n):

    return np.floor(mdates.datestr2num(df['recorded'][n]))



def get_Active(df, n):

    return (df['confirmed'][n]-df['recovered'][n]-df['deaths'][n])



def get_Resolved(df, n):

    return (df['recovered'][n]+df['deaths'][n])         



def transform_data(df2):

    df = df2.copy()



    df.reset_index(inplace=True, drop = True)

    

    df['deaths'] = df['deaths'].fillna(0.0)

    df['confirmed']= df['confirmed'].fillna(0.0)

    df['recovered']= df['recovered'].fillna(0.0)

    df['province'] = df['province'].fillna('')



    df = add_column('calc_active', df, get_Active)

    df = add_column('resolved', df, get_Resolved)

    df = add_column('mDate', df, get_mDate)



    # Strip string values and fill na's

    df['country']=list(map(lambda n: n.strip(), df['country']))

    df['province']=df['province'].fillna('')

    df['city']=df['city'].fillna('')



    #Clarifications needed as some countries are referred to by more than one identifier

    #  there are probably more edits I could make, but I'm ok with this for now.

    df.replace(to_replace='Mainland China',value='China', inplace=True)

    df['country'].replace(to_replace='Hong Kong', value = 'China', inplace=True)

    df.replace(to_replace="""('St. Martin',)""",value='St. Martin', inplace=True)

    df.replace(to_replace="Bahamas, The",value='Bahamas', inplace=True)

    df.replace(to_replace="The Bahamas",value='Bahamas', inplace=True)

    df.replace(to_replace="Gambia, The",value='Gambia', inplace=True)

    df.replace(to_replace="The Gambia",value='Gambia', inplace=True)

    df.replace(to_replace="Republic of Ireland",value='Ireland', inplace=True)

    df.replace(to_replace="UK",value='United Kingdom', inplace=True)

    df.replace(to_replace="Congo (Brazzaville)",value='Republic of the Congo', inplace=True)

    df.replace(to_replace="Congo (Kinshasa)",value='Democratic Republic of the Congo', inplace=True)

#     logger.info('transform_data() - returning df\n'+str(df.values))

    # drop provinces with non-sensical names

    df.drop(df[df['province'] == 'Unassigned Location, WA'].index, axis = 0, inplace = True)

    df.drop(df[df['province'] == 'Unknown Location, MA'].index, axis = 0, inplace = True)

    df.drop(df[df['country'] == 'Viet Nam'].index, axis = 0, inplace = True)

    df.drop(df[df['province'] == 'Recovered'].index, axis = 0, inplace = True)

    df.drop(df[df['province'] == 'None'].index, axis = 0, inplace = True)

    df.drop(df[df['province'] == 'Unknown'].index, axis = 0, inplace = True)

    df.drop(df[df['country'] == 'Russian Federation'].index, axis = 0, inplace = True)

    df.drop(df[df['country'] == 'Others'].index, axis = 0, inplace = True)

    # roll Denmark, Denmark into '', Denmark, etc.

    flt = (df['country'] == 'Denmark') & (df['province'] == 'Denmark')

    df.loc[df[flt].index, 'province'] = df.loc[df[flt].index, 'province'].replace(to_replace = 'Denmark', value = '')

    flt = (df['country'] == 'France') & (df['province'] == 'France')

    df.loc[df[flt].index, 'province'] = df.loc[df[flt].index, 'province'].replace(to_replace = 'France', value = '')

    flt = (df['country'] == 'Mexico') & (df['province'] == 'Mexico')

    df.loc[df[flt].index, 'province'] = df.loc[df[flt].index, 'province'].replace(to_replace = 'Mexico', value = '')

    flt = (df['country'] == 'Netherlands') & (df['province'] == 'Netherlands')

    df.loc[df[flt].index, 'province'] = df.loc[df[flt].index, 'province'].replace(to_replace = 'Netherlands', value = '')

    flt = (df['country'] == 'United Kingdom') & (df['province'] == 'United Kingdom')

    df.loc[df[flt].index, 'province'] = df.loc[df[flt].index, 'province'].replace(to_replace = 'United Kingdom', value = '')

    flt = (df['country'] == 'US') & (df['province'] == 'Chicago')

    df.loc[df[flt].index, 'province'] = df.loc[df[flt].index, 'province'].replace(to_replace = 'Chicago', value = 'Chicago, IL')

    flt = (df['country'] == 'Macau') | (df['province'] == 'Macau') 

    df.loc[df[flt].index, 'country'] = df.loc[df[flt].index, 'country'].replace({'Macau':'China', '':'China'})

    flt = (df['country'] == 'Taiwan') | (df['province'] == 'Taiwan')

    df.loc[df[flt].index, 'country'] = df.loc[df[flt].index, 'country'].replace({'', 'Taiwan'})

    df.loc[df[flt].index, 'province'] = df.loc[df[flt].index, 'province'].replace({'Taiwan', ''})

    flt = (df['country'] == 'Taiwan*')

    df.loc[df[flt].index, 'country'] = df.loc[df[flt].index, 'country'].replace({'Taiwan*', 'Taiwan'})    

    # drop 4 rows for US, US only because they don't fit into part of a larger sequence

    flt = (df['country'] == 'US') & (df['province'] == 'US')

    df.drop(df[flt].index, axis = 0, inplace = True)

    return df         



# Declare universal variables



# url = '../input//pull-datacsv//05-20-2020.csv'

# url = '../input//pull-datacsv//06-02-2020.csv'

# url = '../input//pull-datacsv//06-06-2020.csv'

url = '../input//pull-datacsv//06-28-2020.csv'



df = pull_data(url)

df = transform_data(df)
def getMortRate(df):

    deaths = df['deaths'].to_numpy()[-1]

    cases = df['confirmed'].to_numpy()[-1]

    lastdate = df.index.to_numpy()[-1] - DaysToResolve

    df2 = df[df.index <= lastdate]

    if df2.shape[0]>0:

        cases = df2['confirmed'].to_numpy()[-1]

    if deaths > cases:

        deaths = cases

    if cases != 0:

        return np.abs(100*deaths/cases)

    else:

        return np.NaN



def PlotCases(df, sTitle):

    fig = pyplot.figure(dpi=150)

    plt = fig.add_subplot(1,1,1)

    ax1 = plt.plot(df['mDate'], df['confirmed'],label='All Cases', color='b')

    ax2 = plt.plot(df['mDate'], df['active'], label = 'Active Cases', color = 'orange')

    ax3 = plt.plot(df['mDate'], df['resolved'], label = 'Resolved Cases', color='g')

    ax4 = plt.twinx()

    ax4.fill_between(df['mDate'], 0, df['deaths'], label = 'Deaths', color='r', alpha = 0.2)

    if (df['deaths'].max() == np.NaN) or (df['deaths'].max() == 0):

        deaths = 5.0

    else:

        deaths = df['deaths'].max()  

    ax4.set_ylim(0, 2*deaths)

    ax4.grid(False)

    xMortRate = getMortRate(df)

    if xMortRate != np.nan and xMortRate > 0.0:

        xPos = (df['mDate'].to_numpy()[-1]+df['mDate'].to_numpy()[0])/2

        yPos = (df['confirmed'].max())/10

        plt.text(xPos, yPos, 'Mortality Rate: '+"{:.2f}".format(xMortRate)+'%',\

                verticalalignment = 'bottom', horizontalalignment = 'center',\

                color = 'k', fontsize=7, alpha=0.6)

    plt.grid(True)

    leg = plt.legend(loc=2,ncol=1,prop={'size':7})

    leg.get_frame().set_alpha(0.4)

    leg = ax4.legend(loc=6,ncol=1,prop={'size':7})

    leg.get_frame().set_alpha(0.4)

    plt.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    for label in plt.xaxis.get_ticklabels():

        label.set_rotation(45)

    plt.set_title(sTitle)

    



def getPlotData(df, country, province = '', city = '', ):

    if len(city) > 0:

        flt = (df['country']==country) & (df['province']==province) & (df['city']==city)

        dfint = df[flt].loc[:,['mDate', 'confirmed', 'calc_active', 'deaths', 'resolved']]

        dfint['mDate']=np.floor(dfint.loc[:, 'mDate'])

        dfint.drop_duplicates(subset = 'mDate', inplace = True)

        return dfint.groupby('mDate').agg(\

                    confirmed = ('confirmed', ['first']),\

                    active = ('calc_active', 'first'),\

                    deaths = ('deaths', 'first'),\

                    resolved = ('resolved', 'sum'))

    elif (len(province) >0):

        flt = ((df['country']==country) & (df['province']==province))

        dfint = df[flt].loc[:,['city', 'mDate', 'confirmed', 'calc_active', 'deaths', 'resolved']]

        dfint['mDate'] = np.floor(dfint.loc[:, 'mDate'])

        dfint.drop_duplicates(subset = ['mDate', 'city'], inplace = True)

        return dfint.groupby('mDate').agg(\

                    confirmed = ('confirmed', 'sum'),\

                    active = ('calc_active', 'sum'),\

                    deaths = ('deaths', 'sum'),\

                    resolved = ('resolved', 'sum'))

    else:

        flt = (df['country']==country)

        dfint = df[flt].loc[:,['province', 'city', 'mDate', 'confirmed', 'calc_active', 'deaths', 'resolved']]

        dfint['mDate'] = np.floor(dfint.loc[:, 'mDate'])

        dfint.drop_duplicates(subset = ['mDate', 'province', 'city'], inplace = True)

        return dfint.groupby('mDate').agg(\

                    confirmed = ('confirmed', 'sum'),\

                    active = ('calc_active', 'sum'),\

                    deaths = ('deaths', 'sum'),\

                    resolved = ('resolved', 'sum'))



DaysToResolve = 19

MinPlotPoints = 25



#Execution

for country in df.groupby('country'):

    sCountry = country[0].strip()

    df2 = getPlotData(df, country = sCountry)

    df2.reset_index(inplace=True)

#     print(df2)

    if df2.shape[0]>MinPlotPoints:

        PlotCases(df2, sTitle= sCountry)

    for province in df[df['country']==country[0]].groupby('province'):

        sProvince = province[0].strip()

        if len(sProvince) != 0:

            df2 = getPlotData(df, country= sCountry, province = sProvince)

            df2.reset_index(inplace=True)

            if df2.shape[0]>MinPlotPoints:

                PlotCases(df2, sTitle = sProvince+', '+sCountry)
def populateRegions(dfIn):

    dfReturn = dfIn.loc[:,['province', 'country']]

#    Some countries in the list are only represented by their provinces, so it is necessary to append a list of countries

    dfAppend = dfIn.loc[:, ['country']]

    dfAppend['province'] = ''

    dfReturn = dfReturn.append(dfAppend)

#     dfReturn['country']=list(map(lambda name: name.strip(), dfReturn['country']))

#     dfReturn['province']=list(map(lambda name: name.strip(), dfReturn['province']))

    dfReturn = dfReturn.drop_duplicates().sort_values(by= ['country', 'province'], ascending=True)

    dfReturn.reset_index(inplace = True, drop = True)

    return dfReturn



dfRegionData = populateRegions(df)

dfRegionData
def getDays2Resolve(df, country, province = ''):

    dfSeries = getPlotData(df, country, province)

    dfSeries.reset_index(inplace = True)

    #cycle down from the end of the series locating the date of the 

    #   lowest number of confirmed that is greater than resolved

    resolvedays = []

    for n in dfSeries.index[-1::-1]:

        res = dfSeries['resolved'][n]

        flt = dfSeries['confirmed']>=res

        if dfSeries[flt]['mDate'].shape[0]>0:

            dtconfirmed = dfSeries[flt]['mDate'].min()

            resolvedays.append(dfSeries['mDate'][n]-dtconfirmed)

    nresolve = np.array(resolvedays)

    if len(nresolve)>0:

        return nresolve.mean(), nresolve.std(), len(nresolve)

    else:

        return np.NaN, np.NaN, np.NaN



daysToRecover = []

for n in dfRegionData.index:

    prov = str(dfRegionData['province'][n])

    if prov != 'nan' and len(prov)>0:

        daysToRecover.append(getDays2Resolve(df, dfRegionData['country'][n], dfRegionData['province'][n]))

    else:

        daysToRecover.append(getDays2Resolve(df, country = dfRegionData['country'][n]))

nparr = np.array(daysToRecover)

dfRegionData['resolveDaysMean'] = nparr[:,0]

dfRegionData['resolveDaysStd'] = nparr[:,1]

dfRegionData['resolvePoints'] = nparr[:,2]

dfRegionData
dfPlotData = dfRegionData

flt = (dfPlotData['resolveDaysMean']>0.0) & (dfPlotData['resolveDaysStd']>0.0)

dfPlotData = dfPlotData[flt]

dfPlotData['ratio']=list(map(lambda x,y:x/y*100, dfPlotData['resolveDaysStd'],dfPlotData['resolveDaysMean']))

flt = dfPlotData['ratio']<140.0

dfPlotData = dfPlotData[flt]

fig = pyplot.figure(dpi=150, figsize = (8,8))



gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),

                      left=0.1, right=0.9, bottom=0.1, top=0.9,

                      wspace=0.05, hspace=0.05)



ax = fig.add_subplot(gs[1, 0])

ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)

ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

ax.scatter(dfPlotData.loc[:,('resolveDaysMean')],dfPlotData.loc[:,('ratio')], s = 1)

ax2 = ax_histx.hist(dfPlotData['resolveDaysMean'], bins = 100, rwidth = 0.8)

ax3 = ax_histy.hist(dfPlotData['ratio'], bins = 100, rwidth =0.8, orientation = 'horizontal')

ax.grid(True)

ax_histx.grid(True)

ax_histy.grid(True)

ax.set_ylabel('Std. Dev. as a Percentage')

ax.set_xlabel('Days to Resolve')

ax_histx.set_title('Days to Resolve Covid-19 by Region')

font_dict = {'family':'serif',

                'color':'darkred',

                'size':8}

ax.text(40,80,'One region somewhere on earth represented by one marker.', fontdict = font_dict, ha = 'center', wrap = True)

fig.show()
def linreg(df):

    """Returns linear regression data on first two columns of the indexed pandas Dataframe passed to it"""  

#     print(df)

    n = df.shape[0]

    sumxy, sumx, sumx2, sumy, sumy2, sumvarx, sumvary, sumvarxvary, sumvarx2, sumvary2 = 10 * [0.0, ]

    meanx, meany = df.mean()[0], df.mean()[1]

    for i in range(n):

        x = df.iloc[i,0]

        y = df.iloc[i,1]

#         print(x)

        sumxy += x * y

        sumx += x

        sumx2 += x ** 2

        sumy += y

        sumy2 += y ** 2

        sumvarx += x-meanx

        sumvary += y-meany

        sumvarxvary += sumvarx*sumvary

        sumvarx2 += np.square(sumvarx)

        sumvary2 += np.square(sumvary)

    m = (n*sumxy-sumx*sumy)/(n*sumx2 - sumx**2)

    b = (sumy*sumx2-sumx*sumxy)/(n*sumx2 - sumx**2)

    r = (sumvarxvary)/(sumvarx2 * sumvary2) ** 0.5

    return m, b, r



def get_Series(df, country, province = ''):

    dfint = getPlotData(df, country, province)

    # First, enumerate the date, value pairs by creating a dataframe with an index 

    dfint.reset_index(inplace = True)

    return dfint.loc[:, ['mDate', 'confirmed']]



def get_BestFitExp(df, country, province = ''):

    dfint = get_Series(df, country, province)

    #append daily growth rates

    growth = [0,]

    for n in range(dfint.shape[0]-1):

        timedif = dfint.iloc[n+1,0]-dfint.iloc[n,0]

        if timedif > 0:

            growth.append(dfint.iloc[n+1,1]/(dfint.iloc[n,1]*timedif))

        else:

            growth.append(np.NaN)

    dfint['dgrowth'] = growth

#     print(dfint)

    #determine range of greatest interest to reduce calculation time

    dfsorted = dfint.sort_values(by='dgrowth', ascending = False)

    dfsorted.reset_index(inplace=True)

#     print(dfsorted)

    cutoff = 0.1

    growthcutoff = dfsorted.loc[np.floor(dfsorted.shape[0]*cutoff), 'dgrowth']

#     print('Growth cut-off value is %s' % growthcutoff)

    flt = dfint['dgrowth'] >= growthcutoff

    dfsorted = dfint[flt]

#     print(dfsorted)

    highDate = dfsorted.agg({'mDate': 'max'})[0]

    lowDate = dfsorted.agg({'mDate': 'min'})[0]    

    #insert logarithm of 'confirmed'

    def lbind(value, cutoff):

        if value < cutoff:

            return cutoff

        else:

            return value

    dfint.insert(1, 'lnconfirmed', list(map(lambda value: np.log(lbind(value, np.exp(-100))), dfint['confirmed'])))

    #iterate through the series trimming leading and trailing dates, accepting each trim if r increases as a result

    #Grab the index of lowDate and highDate

    lowIndex = dfint.loc[dfint['mDate'] == lowDate].index[0]

    highIndex = dfint[(dfint['mDate'] == highDate)].index[0]

#     print('Starting lowIndex = %s and starting highIndex = %s' % (lowIndex, highIndex))

    #Define the minimum distance between lowIndex and highIndex

    minDist = 7

    maxIter = 100

    stopChecking = False

    bestR = linreg(dfint.loc[lowIndex:(highIndex+1), ['mDate', 'lnconfirmed']])  

    while (highIndex - lowIndex > minDist) & (not stopChecking) & (maxIter > 0):

        maxIter -= 1

        bestR = linreg(dfint.loc[lowIndex:(highIndex+1), ['mDate', 'lnconfirmed']])        

        if highIndex - lowIndex > minDist:

            rLowInside = linreg(dfint.loc[(lowIndex+1):(highIndex+1), ['mDate', 'lnconfirmed']])

            rHighInside = linreg(dfint.loc[(lowIndex):(highIndex), ['mDate', 'lnconfirmed']])

        else:

            rLowInside = (0, 0, 0)

            rHighInside = (0, 0, 0)

        if highIndex < (dfint.shape[0]-1):

            rHighOutside = linreg(dfint.loc[(lowIndex):(highIndex+2), ['mDate', 'lnconfirmed']])

        else:

            rHighOutside = (0,0,0)

        if lowIndex > 0:

            rLowOutside = linreg(dfint.loc[(lowIndex-1):(highIndex+1), ['mDate', 'lnconfirmed']])

        else:

            rLowOutside = (0,0,0)

        bestmove = np.max([(bestR[0]*bestR[2]), (rLowInside[0]*rLowInside[2]), \

                            (rHighInside[0]*rHighInside[2]), (rHighOutside[0]*rHighOutside[2]),\

                           (rLowOutside[0]*rLowOutside[2])])

        if bestmove == (bestR[0]*bestR[2]): #Stay put, most optimal solution has been achieved

            stopChecking = True

        elif bestmove == (rLowInside[0]*rLowInside[2]): #Increase lowIndex

            lowIndex += 1

#             print('Increasing lowIndex to %s' % lowIndex)

        elif bestmove == (rHighInside[0]*rHighInside[2]): #Decrease highIndex

            highIndex -= 1

#             print('Decreasing highIndex to %s' % highIndex)

        elif bestmove == (rHighOutside[0]*rHighOutside[2]): #Increase highIndex

            highIndex += 1

#             print('Increasing highIndex to %s' % highIndex)

        elif bestmove == (rLowOutside[0]*rLowOutside[2]): #Decrease lowIndex

            lowIndex -= 1

#             print('Decreasing lowIndex to %s' % lowIndex)

    bestR = list(bestR)

    bestR.append(lowIndex)

    bestR.append(highIndex)

    return bestR



expprog = []

for n in dfRegionData.index:

    province, country = dfRegionData.loc[n, ['province', 'country']]

    expprog.append(get_BestFitExp(df, country, province))

expprog = np.array(expprog)

dfRegionData['expFit_k'] = list(map(lambda value: np.exp(value), expprog[:,0]))

dfRegionData['expFit_t0'] = list(map(lambda a, b: -a/np.maximum(b,0.000001), expprog[:,1], expprog[:,0]))

dfRegionData['expFit_r'] = expprog[:,2]

dfRegionData['expFit_lowNdx'] = expprog[:,3]

dfRegionData['expFit_highNdx'] = expprog[:,4]



dfRegionData.style.background_gradient(cmap='Greens')